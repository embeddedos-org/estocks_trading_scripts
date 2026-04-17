"""
Interactive Brokers Connection Factory
=======================================

Provides a factory pattern for creating IB connections using either
ib_async (recommended) or the native ibapi library.

Usage:
    conn = IBConnection.create(backend="ib_async", host="127.0.0.1", port=7497)
    async with conn:
        # use connection
        pass
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode — paper or live."""
    PAPER = "paper"
    LIVE = "live"


@dataclass
class ConnectionConfig:
    """Configuration for an IB connection."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: float = 30.0
    readonly: bool = False
    auto_reconnect: bool = True
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10

    @property
    def trading_mode(self) -> TradingMode:
        """Determine trading mode from port number.

        Port 7497 = TWS Paper, 7496 = TWS Live,
        4002 = Gateway Paper, 4001 = Gateway Live.
        """
        if self.port in (7497, 4002):
            return TradingMode.PAPER
        return TradingMode.LIVE


class BaseConnection(ABC):
    """Abstract base class for IB connections."""

    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config
        self._connected = False
        self._reconnect_count = 0

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to IB Gateway/TWS."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from IB Gateway/TWS."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active."""

    @property
    def trading_mode(self) -> TradingMode:
        return self.config.trading_mode

    def __enter__(self) -> "BaseConnection":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()

    async def __aenter__(self) -> "BaseConnection":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()


class IBAsyncConnection(BaseConnection):
    """Connection implementation using the ib_async library (recommended).

    ib_async provides a Pythonic, async-friendly wrapper around the IB API
    with automatic message handling and event-driven architecture.
    """

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._ib: Any = None
        self._import_ib_async()

    def _import_ib_async(self) -> None:
        """Lazily import ib_async with graceful error handling."""
        try:
            import ib_async
            self._ib_module = ib_async
            self._ib = ib_async.IB()
        except ImportError:
            raise ImportError(
                "ib_async is required for IBAsyncConnection. "
                "Install it with: pip install ib-async"
            )

    @property
    def ib(self) -> Any:
        """Access the underlying ib_async.IB instance."""
        return self._ib

    async def connect(self) -> None:
        """Connect to IB Gateway/TWS using ib_async."""
        if self.is_connected():
            logger.warning("Already connected to IB")
            return

        mode = self.config.trading_mode
        logger.info(
            "Connecting to IB via ib_async [%s:%d, client=%d, mode=%s]",
            self.config.host, self.config.port,
            self.config.client_id, mode.value,
        )

        try:
            await self._ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly,
            )
            self._connected = True
            self._reconnect_count = 0

            if self.config.auto_reconnect:
                self._setup_auto_reconnect()

            logger.info(
                "Connected to IB [account=%s, mode=%s]",
                self._get_account_id(), mode.value,
            )
        except Exception as e:
            logger.error("Failed to connect to IB: %s", e)
            raise ConnectionError(f"IB connection failed: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from IB Gateway/TWS."""
        if self._ib and self.is_connected():
            logger.info("Disconnecting from IB")
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        """Check if ib_async connection is active."""
        return self._ib is not None and self._ib.isConnected()

    def _get_account_id(self) -> str:
        """Get the primary account ID."""
        accounts = self._ib.managedAccounts()
        return accounts[0] if accounts else "unknown"

    def _setup_auto_reconnect(self) -> None:
        """Register auto-reconnect handler on disconnect events."""
        def on_disconnected() -> None:
            if not self.config.auto_reconnect:
                return
            if self._reconnect_count >= self.config.max_reconnect_attempts:
                logger.error(
                    "Max reconnect attempts (%d) reached. Giving up.",
                    self.config.max_reconnect_attempts,
                )
                return

            self._reconnect_count += 1
            wait = self.config.reconnect_interval * self._reconnect_count
            logger.warning(
                "Disconnected from IB. Reconnecting in %.1fs (attempt %d/%d)...",
                wait, self._reconnect_count, self.config.max_reconnect_attempts,
            )
            asyncio.ensure_future(self._reconnect_after(wait))

        self._ib.disconnectedEvent += on_disconnected

    async def _reconnect_after(self, wait: float) -> None:
        """Wait and then attempt to reconnect."""
        await asyncio.sleep(wait)
        try:
            await self.connect()
        except ConnectionError:
            logger.error("Reconnection attempt %d failed", self._reconnect_count)

    def reqContractDetails(self, contract: Any) -> Any:
        """Request contract details (passthrough to ib_async)."""
        return self._ib.reqContractDetails(contract)

    def qualifyContracts(self, *contracts: Any) -> Any:
        """Qualify contracts (passthrough to ib_async)."""
        return self._ib.qualifyContracts(*contracts)

    def placeOrder(self, contract: Any, order: Any) -> Any:
        """Place an order (passthrough to ib_async)."""
        trade = self._ib.placeOrder(contract, order)
        logger.info("Order placed: %s %s", contract.symbol, order.action)
        return trade

    def cancelOrder(self, order: Any) -> Any:
        """Cancel an order (passthrough to ib_async)."""
        return self._ib.cancelOrder(order)

    def reqHistoricalData(self, contract: Any, **kwargs: Any) -> Any:
        """Request historical data (passthrough to ib_async)."""
        return self._ib.reqHistoricalData(contract, **kwargs)

    def reqMktData(self, contract: Any, **kwargs: Any) -> Any:
        """Request market data (passthrough to ib_async)."""
        return self._ib.reqMktData(contract, **kwargs)

    def portfolio(self) -> list:
        """Get portfolio items."""
        return self._ib.portfolio()

    def positions(self) -> list:
        """Get positions."""
        return self._ib.positions()

    def accountSummary(self) -> list:
        """Get account summary."""
        return self._ib.accountSummary()


class IBApiConnection(BaseConnection):
    """Connection implementation using the native ibapi library.

    Uses the thread-based approach with EWrapper/EClient pattern
    from the official IB API.
    """

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._app: Any = None
        self._thread: Optional[threading.Thread] = None
        self._next_order_id: int = 0
        self._import_ibapi()

    def _import_ibapi(self) -> None:
        """Lazily import ibapi with graceful error handling."""
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper

            class IBApp(EWrapper, EClient):
                """Combined EWrapper/EClient application."""

                def __init__(self, connection: IBApiConnection) -> None:
                    EClient.__init__(self, self)
                    self._connection = connection
                    self.next_valid_id: int = 0
                    self._data: dict = {}
                    self._errors: list = []

                def nextValidId(self, orderId: int) -> None:
                    self.next_valid_id = orderId
                    self._connection._next_order_id = orderId
                    logger.info("Next valid order ID: %d", orderId)

                def error(self, reqId: int, errorCode: int, errorString: str,
                          advancedOrderRejectJson: str = "") -> None:
                    if errorCode in (2104, 2106, 2158):
                        logger.info("IB info [%d]: %s", errorCode, errorString)
                    else:
                        logger.error(
                            "IB error [reqId=%d, code=%d]: %s",
                            reqId, errorCode, errorString,
                        )
                        self._errors.append({
                            "reqId": reqId,
                            "code": errorCode,
                            "message": errorString,
                        })

                def connectAck(self) -> None:
                    logger.info("IB API connection acknowledged")

            self._app_class = IBApp
        except ImportError:
            raise ImportError(
                "ibapi is required for IBApiConnection. "
                "Install the official IB API: pip install ibapi"
            )

    def connect(self) -> None:
        """Connect to IB Gateway/TWS using the native ibapi."""
        if self.is_connected():
            logger.warning("Already connected to IB via ibapi")
            return

        logger.info(
            "Connecting to IB via ibapi [%s:%d, client=%d]",
            self.config.host, self.config.port, self.config.client_id,
        )

        try:
            self._app = self._app_class(self)
            self._app.connect(
                self.config.host,
                self.config.port,
                self.config.client_id,
            )

            self._thread = threading.Thread(
                target=self._app.run,
                daemon=True,
                name="ib-api-thread",
            )
            self._thread.start()

            # Wait for connection acknowledgment
            deadline = time.time() + self.config.timeout
            while self._app.next_valid_id == 0 and time.time() < deadline:
                time.sleep(0.1)

            if self._app.next_valid_id == 0:
                raise ConnectionError("Timed out waiting for IB API connection")

            self._connected = True
            logger.info("Connected to IB via ibapi")
        except Exception as e:
            logger.error("Failed to connect via ibapi: %s", e)
            raise ConnectionError(f"IB API connection failed: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from IB Gateway/TWS."""
        if self._app and self.is_connected():
            logger.info("Disconnecting from IB via ibapi")
            self._app.disconnect()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            self._connected = False
            logger.info("Disconnected from IB via ibapi")

    def is_connected(self) -> bool:
        """Check if ibapi connection is active."""
        return (
            self._app is not None
            and self._connected
            and self._thread is not None
            and self._thread.is_alive()
        )

    def get_next_order_id(self) -> int:
        """Get the next valid order ID and increment."""
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    @property
    def app(self) -> Any:
        """Access the underlying IBApp (EWrapper/EClient) instance."""
        return self._app


class IBConnection:
    """Factory class for creating IB connections.

    Usage:
        # ib_async backend (recommended)
        conn = IBConnection.create(backend="ib_async", port=7497)

        # Native ibapi backend
        conn = IBConnection.create(backend="ibapi", port=7497)

        # Legacy alias
        conn = IBConnection.create(backend="ib_insync", port=7497)

        # With context manager
        async with IBConnection.create() as conn:
            # use connection
            pass
    """

    BACKENDS = {
        "ib_async": IBAsyncConnection,
        "ib_insync": IBAsyncConnection,  # legacy alias
        "ibapi": IBApiConnection,
    }

    @staticmethod
    def create(
        backend: str = "ib_async",
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: float = 30.0,
        readonly: bool = False,
        auto_reconnect: bool = True,
    ) -> BaseConnection:
        """Create a new IB connection using the specified backend.

        Args:
            backend: Connection library to use ("ib_async", "ib_insync" [legacy alias], or "ibapi").
            host: IB Gateway/TWS hostname.
            port: IB Gateway/TWS port (7497=TWS Paper, 7496=TWS Live,
                  4002=Gateway Paper, 4001=Gateway Live).
            client_id: Unique client identifier.
            timeout: Connection timeout in seconds.
            readonly: If True, connect in read-only mode (no order placement).
            auto_reconnect: If True, automatically reconnect on disconnect.

        Returns:
            A BaseConnection instance for the specified backend.

        Raises:
            ValueError: If an unknown backend is specified.
            ImportError: If the required library is not installed.
        """
        if backend not in IBConnection.BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available: {list(IBConnection.BACKENDS.keys())}"
            )

        config = ConnectionConfig(
            host=host,
            port=port,
            client_id=client_id,
            timeout=timeout,
            readonly=readonly,
            auto_reconnect=auto_reconnect,
        )

        connection_class = IBConnection.BACKENDS[backend]
        logger.info(
            "Creating %s connection [%s:%d, mode=%s]",
            backend, host, port, config.trading_mode.value,
        )
        return connection_class(config)
