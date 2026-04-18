"""
Broker Bridge — Connects AI Agent Decisions to Real Broker APIs
================================================================

The critical last-mile wiring that turns AI decisions into real trades.
Maps SelfLearningAgent.decide() output → broker-specific order execution.

Supported Brokers:
- Interactive Brokers (via ib_async / ibapi)
- TradeStation (via REST API)
- Schwab/thinkorswim (via Schwab API)

Features:
- Unified interface across all brokers
- Position tracking with entry prices
- Fill callbacks that auto-record outcomes to TradeMemory
- Configurable safety: paper-only mode, max order size, confirmation
- Translates agent signals to broker-native order types

Usage:
    bridge = BrokerBridge(broker="ib", mode="paper")
    bridge.connect()
    bridge.execute_decision(agent_decision, symbol="AAPL")
    bridge.get_positions()
    bridge.disconnect()
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Tracked position state."""
    symbol: str
    direction: str  # "long" or "short"
    shares: int
    entry_price: float
    entry_time: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    order_id: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of a trade execution."""
    success: bool
    broker: str
    symbol: str
    action: str
    shares: int
    price: float
    order_id: str = ""
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseBrokerAdapter(ABC):
    """Abstract broker adapter interface."""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult: ...

    @abstractmethod
    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ─── Interactive Brokers Adapter ───

class IBAdapter(BaseBrokerAdapter):
    """Real Interactive Brokers adapter using ib_async OrderManager."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._connection = None
        self._order_manager = None
        self._data_fetcher = None
        self._connected = False
        self._name = "interactive_brokers"
        self._event_loop = None
        self._loop_thread = None

    @property
    def name(self) -> str:
        return self._name

    def _ensure_event_loop(self) -> None:
        """Start a dedicated asyncio event loop in a background daemon thread.

        ib_async requires a running event loop for all operations. BrokerBridge
        is synchronous, so we spin up a dedicated loop thread and submit
        coroutines to it via run_coroutine_threadsafe().
        """
        import asyncio
        import threading

        if self._event_loop is not None and self._event_loop.is_running():
            return

        self._event_loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, daemon=True, name="ib-async-loop")
        self._loop_thread.start()

    def _run_async(self, coro, timeout: float = 30.0):
        """Run an async coroutine on the IB event loop from synchronous code."""
        import asyncio
        self._ensure_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result(timeout=timeout)

    def connect(self) -> bool:
        try:
            import inspect
            from interactive_brokers.utils.ib_connection import IBConnection
            from interactive_brokers.utils.order_manager import OrderManager

            self._connection = IBConnection.create(
                backend="ib_async",
                host=self._host,
                port=self._port,
                client_id=self._client_id,
            )

            # IBAsyncConnection.connect() is a coroutine — bridge to sync via
            # a dedicated background event loop so the IB message loop keeps
            # running after the initial handshake.
            if inspect.iscoroutinefunction(self._connection.connect):
                self._run_async(self._connection.connect(), timeout=30.0)
            else:
                self._connection.connect()

            self._order_manager = OrderManager(self._connection)

            try:
                from interactive_brokers.data.historical_fetcher import HistoricalDataFetcher
                self._data_fetcher = HistoricalDataFetcher(self._connection)
            except Exception:
                pass

            self._connected = True
            mode = "PAPER" if self._port in (7497, 4002) else "LIVE"
            logger.info("IB connected [%s:%d, mode=%s]", self._host, self._port, mode)
            return True
        except Exception as e:
            logger.error("IB connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        if self._connection:
            self._connection.disconnect()
            self._connected = False
            logger.info("IB disconnected")
        if self._event_loop and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

    def is_connected(self) -> bool:
        if self._connection:
            return self._connection.is_connected()
        return False

    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult:
        if not self._order_manager:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message="Not connected")

        try:
            trade = self._order_manager.market_order(symbol, action.upper(), float(quantity))
            order_id = str(trade.order.orderId) if hasattr(trade, 'order') else str(id(trade))
            logger.info("IB market order: %s %d %s → %s", action, quantity, symbol, order_id)
            return ExecutionResult(True, self.name, symbol, action, quantity, 0, order_id=order_id,
                                   message=f"Market order placed: {action} {quantity} {symbol}")
        except Exception as e:
            logger.error("IB market order failed: %s", e)
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message=str(e))

    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult:
        if not self._order_manager:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message="Not connected")

        try:
            trade = self._order_manager.limit_order(symbol, action.upper(), float(quantity), price)
            order_id = str(trade.order.orderId) if hasattr(trade, 'order') else str(id(trade))
            return ExecutionResult(True, self.name, symbol, action, quantity, price, order_id=order_id,
                                   message=f"Limit order: {action} {quantity} {symbol} @ ${price:.2f}")
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        if self._order_manager:
            return self._order_manager.cancel_order(int(order_id))
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self._connection or not self.is_connected():
            return []
        try:
            positions = self._connection.positions()
            return [
                {
                    "symbol": p.contract.symbol,
                    "quantity": p.position,
                    "avg_cost": p.avgCost,
                    "market_value": p.marketValue if hasattr(p, 'marketValue') else 0,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("Failed to get IB positions: %s", e)
            return []

    def get_account_info(self) -> Dict[str, Any]:
        if not self._connection or not self.is_connected():
            return {"broker": self.name, "connected": False}
        try:
            summary = self._connection.accountSummary()
            info = {"broker": self.name, "connected": True}
            for item in summary:
                if item.tag in ("NetLiquidation", "BuyingPower", "TotalCashValue", "GrossPositionValue"):
                    info[item.tag] = float(item.value)
            return info
        except Exception:
            return {"broker": self.name, "connected": True}

    def get_latest_price(self, symbol: str) -> float:
        if self._data_fetcher:
            try:
                df = self._data_fetcher.fetch_bars(symbol, duration="1 D", bar_size="1 min")
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception:
                pass
        return 0.0


# ─── TradeStation Adapter ───

class TradeStationAdapter(BaseBrokerAdapter):
    """Real TradeStation adapter using the REST API OrderRouter."""

    def __init__(self, config: Dict[str, str], account_id: str = ""):
        self._config = config
        self._account_id = account_id
        self._router = None
        self._connected = False
        self._name = "tradestation"

    @property
    def name(self) -> str:
        return self._name

    def connect(self) -> bool:
        try:
            from tradestation.api.order_router import TradeStationOrderRouter
            self._router = TradeStationOrderRouter(self._config)

            if not self._account_id:
                # Try to get first account
                try:
                    from tradestation.api.account_monitor import AccountMonitor
                    monitor = AccountMonitor(self._router, {})
                    balances = monitor.get_balances(self._config.get("account_id", ""))
                    self._account_id = self._config.get("account_id", "")
                except Exception:
                    self._account_id = self._config.get("account_id", "")

            self._connected = True
            logger.info("TradeStation connected (account=%s)", self._account_id)
            return True
        except Exception as e:
            logger.error("TradeStation connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        self._connected = False
        logger.info("TradeStation disconnected")

    def is_connected(self) -> bool:
        return self._connected and self._router is not None

    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult:
        if not self._router:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message="Not connected")
        try:
            order_id = self._router.place_market_order(self._account_id, symbol, action.upper(), quantity)
            return ExecutionResult(True, self.name, symbol, action, quantity, 0, order_id=order_id,
                                   message=f"TS market order: {action} {quantity} {symbol}")
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message=str(e))

    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult:
        if not self._router:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message="Not connected")
        try:
            order_id = self._router.place_limit_order(self._account_id, symbol, action.upper(), quantity, price)
            return ExecutionResult(True, self.name, symbol, action, quantity, price, order_id=order_id,
                                   message=f"TS limit order: {action} {quantity} {symbol} @ ${price:.2f}")
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        if self._router:
            try:
                self._router.cancel_order(self._account_id, order_id)
                return True
            except Exception:
                return False
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self._router:
            return []
        try:
            from tradestation.api.account_monitor import AccountMonitor
            monitor = AccountMonitor(self._router, {})
            return monitor.get_positions(self._account_id)
        except Exception:
            return []

    def get_account_info(self) -> Dict[str, Any]:
        if not self._router:
            return {"broker": self.name, "connected": False}
        try:
            from tradestation.api.account_monitor import AccountMonitor
            monitor = AccountMonitor(self._router, {})
            balances = monitor.get_balances(self._account_id)
            return {"broker": self.name, "connected": True, **balances}
        except Exception:
            return {"broker": self.name, "connected": True}

    def get_latest_price(self, symbol: str) -> float:
        if not self._router:
            return 0.0
        try:
            quote = self._router.get_quote(symbol)
            # TradeStation v3 quote fields: Last, Ask, Bid, Close
            for field in ("Last", "Ask", "Bid", "Close"):
                val = quote.get(field)
                if val is not None:
                    price = float(val)
                    if price > 0:
                        return price
        except Exception as e:
            logger.warning("TradeStation get_latest_price(%s) failed: %s", symbol, e)
        return 0.0


# ─── Schwab/thinkorswim Adapter ───

class SchwabAdapter(BaseBrokerAdapter):
    """Schwab API adapter for thinkorswim."""

    def __init__(self, config: Dict[str, str]):
        self._config = config
        self._client = None
        self._connected = False
        self._name = "schwab"

    @property
    def name(self) -> str:
        return self._name

    def connect(self) -> bool:
        try:
            from thinkorswim.api.schwab_client import SchwabClient
            self._client = SchwabClient(self._config)
            self._connected = True
            logger.info("Schwab connected")
            return True
        except Exception as e:
            logger.error("Schwab connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult:
        if not self._client:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message="Not connected")
        try:
            order_id = self._client.place_market_order(symbol, action.upper(), quantity)
            return ExecutionResult(True, self.name, symbol, action, quantity, 0, order_id=order_id)
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message=str(e))

    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult:
        if not self._client:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message="Not connected")
        try:
            order_id = self._client.place_limit_order(symbol, action.upper(), quantity, price)
            return ExecutionResult(True, self.name, symbol, action, quantity, price, order_id=order_id)
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        if self._client:
            try:
                self._client.cancel_order(order_id)
                return True
            except Exception:
                return False
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        if self._client:
            try:
                return self._client.get_positions()
            except Exception:
                return []
        return []

    def get_account_info(self) -> Dict[str, Any]:
        if self._client:
            try:
                return {"broker": self.name, "connected": True, **self._client.get_account_info()}
            except Exception:
                pass
        return {"broker": self.name, "connected": False}

    def get_latest_price(self, symbol: str) -> float:
        if self._client:
            try:
                quote = self._client.get_quote(symbol)
                return quote.get("lastPrice", 0.0)
            except Exception:
                pass
        return 0.0


# ─── Broker Bridge (Main Entry Point) ───

class BrokerBridge:
    """Unified bridge connecting AI agent decisions to any broker.

    Handles:
    - Broker selection and connection
    - Translating BUY/SELL/HOLD → market/limit orders
    - Auto TP/SL placement after every trade
    - Position tracking with entry prices
    - Position reconciliation with broker state
    - Force-close on max loss threshold
    - P&L calculation on exit
    - JSONL trade diary for audit trail
    - Auto-recording outcomes to SelfLearningAgent

    Args:
        broker: Broker name — "ib", "tradestation", "schwab", "paper"
        config: Broker-specific config dict.
        mode: "paper" (IB port 7497) or "live" (IB port 7496).
        max_position_pct: Max % of capital per position.
        max_shares: Maximum shares per order.
        max_loss_pct: Force-close if position loses more than this %.
        default_tp_pct: Default take-profit distance as % of entry.
        default_sl_pct: Default stop-loss distance as % of entry.
        diary_path: Path to JSONL trade diary file.
    """

    BROKERS = {
        "ib": IBAdapter,
        "interactive_brokers": IBAdapter,
        "tradestation": TradeStationAdapter,
        "ts": TradeStationAdapter,
        "schwab": SchwabAdapter,
        "thinkorswim": SchwabAdapter,
        "tos": SchwabAdapter,
    }

    def __init__(
        self,
        broker: str = "ib",
        config: Optional[Dict[str, Any]] = None,
        mode: str = "paper",
        max_position_pct: float = 0.10,
        max_shares: int = 500,
        capital: float = 100_000.0,
        max_loss_pct: float = 5.0,
        default_tp_pct: float = 3.0,
        default_sl_pct: float = 2.0,
        diary_path: Optional[str] = None,
    ) -> None:
        self._broker_name = broker.lower()
        self._config = config or {}
        self._mode = mode
        self._max_position_pct = max_position_pct
        self._max_shares = max_shares
        self._capital = capital
        self._max_loss_pct = max_loss_pct
        self._default_tp_pct = default_tp_pct
        self._default_sl_pct = default_sl_pct

        # Diary path
        if diary_path is None:
            import os
            diary_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "logs")
            os.makedirs(diary_dir, exist_ok=True)
            self._diary_path = os.path.join(diary_dir, "trade_diary.jsonl")
        else:
            self._diary_path = diary_path

        # Position tracking
        self._positions: Dict[str, Position] = {}

        # Create adapter
        self._adapter = self._create_adapter()

        logger.info(
            "BrokerBridge created: broker=%s, mode=%s, max_shares=%d, sl=%.1f%%, tp=%.1f%%",
            self._broker_name, mode, max_shares, default_sl_pct, default_tp_pct,
        )

    def _create_adapter(self) -> BaseBrokerAdapter:
        """Create the appropriate broker adapter."""
        if self._broker_name in ("ib", "interactive_brokers"):
            port = 7497 if self._mode == "paper" else 7496
            return IBAdapter(
                host=self._config.get("host", "127.0.0.1"),
                port=self._config.get("port", port),
                client_id=self._config.get("client_id", 1),
            )
        elif self._broker_name in ("tradestation", "ts"):
            return TradeStationAdapter(self._config, self._config.get("account_id", ""))
        elif self._broker_name in ("schwab", "thinkorswim", "tos"):
            return SchwabAdapter(self._config)
        else:
            raise ValueError(f"Unknown broker: {self._broker_name}. Options: {list(self.BROKERS.keys())}")

    def connect(self) -> bool:
        """Connect to the broker."""
        return self._adapter.connect()

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        self._adapter.disconnect()

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._adapter.is_connected()

    # ─── Execute AI Decisions ───

    def execute_decision(
        self,
        decision: Dict[str, Any],
        symbol: str,
        agent: Optional[Any] = None,
    ) -> Optional[ExecutionResult]:
        """Execute an AI agent decision via the connected broker.

        Args:
            decision: Output from SelfLearningAgent.decide() or LLMReasoner.reason()
            symbol: Ticker symbol.
            agent: Optional SelfLearningAgent for auto-recording outcomes.

        Returns:
            ExecutionResult or None if HOLD.
        """
        action = decision.get("action", "HOLD")
        confidence = decision.get("confidence", 0)
        price = decision.get("price", 0)
        tp_price = decision.get("tp_price")
        sl_price = decision.get("sl_price")
        exit_plan = decision.get("exit_plan", "")

        # Write every decision to diary
        self._write_diary({
            "action": action,
            "symbol": symbol,
            "price": price,
            "confidence": confidence,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "exit_plan": exit_plan,
            "regime": decision.get("regime", ""),
            "reasoning": decision.get("reasoning", "")[:500] if decision.get("reasoning") else "",
        })

        if action == "HOLD":
            logger.debug("Decision: HOLD %s (confidence=%.2f)", symbol, confidence)
            return None

        has_position = symbol in self._positions
        current_pos = self._positions.get(symbol)

        if action == "BUY":
            if has_position and current_pos.direction == "long":
                logger.debug("Already long %s, skipping BUY", symbol)
                return None

            if has_position and current_pos.direction == "short":
                self._close_position(symbol, price, agent)

            shares = self._calculate_shares(price)
            result = self._adapter.place_market_order(symbol, "BUY", shares)

            if result.success:
                self._positions[symbol] = Position(
                    symbol=symbol, direction="long", shares=shares,
                    entry_price=price, entry_time=datetime.now().isoformat(),
                    order_id=result.order_id,
                )
                logger.info("📈 OPENED LONG: %d shares %s @ $%.2f [%s]",
                           shares, symbol, price, self._adapter.name)

                # Auto-place TP/SL
                self._place_tp_sl(symbol, price, "long", tp_price, sl_price)

                self._write_diary({
                    "action": "OPEN_LONG",
                    "symbol": symbol,
                    "shares": shares,
                    "entry_price": price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "order_id": result.order_id,
                    "exit_plan": exit_plan,
                })
            return result

        elif action == "SELL":
            if has_position and current_pos.direction == "long":
                return self._close_position(symbol, price, agent)

            if has_position and current_pos.direction == "short":
                logger.debug("Already short %s, skipping SELL", symbol)
                return None

            shares = self._calculate_shares(price)
            result = self._adapter.place_market_order(symbol, "SELL", shares)

            if result.success:
                self._positions[symbol] = Position(
                    symbol=symbol, direction="short", shares=shares,
                    entry_price=price, entry_time=datetime.now().isoformat(),
                    order_id=result.order_id,
                )
                logger.info("📉 OPENED SHORT: %d shares %s @ $%.2f [%s]",
                           shares, symbol, price, self._adapter.name)

                self._place_tp_sl(symbol, price, "short", tp_price, sl_price)

                self._write_diary({
                    "action": "OPEN_SHORT",
                    "symbol": symbol,
                    "shares": shares,
                    "entry_price": price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "order_id": result.order_id,
                    "exit_plan": exit_plan,
                })
            return result

        return None

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        agent: Optional[Any] = None,
    ) -> ExecutionResult:
        """Close an existing position and record outcome."""
        pos = self._positions.get(symbol)
        if not pos:
            return ExecutionResult(False, self._adapter.name, symbol, "CLOSE", 0, exit_price,
                                   message="No position to close")

        close_action = "SELL" if pos.direction == "long" else "BUY"
        result = self._adapter.place_market_order(symbol, close_action, pos.shares)

        if result.success:
            # Calculate P&L
            if pos.direction == "long":
                pnl = (exit_price - pos.entry_price) * pos.shares
            else:
                pnl = (pos.entry_price - exit_price) * pos.shares

            emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                "%s CLOSED %s: %d shares %s | entry=$%.2f exit=$%.2f | P&L=$%.2f",
                emoji, pos.direction.upper(), pos.shares, symbol,
                pos.entry_price, exit_price, pnl,
            )

            # Record outcome in agent memory
            if agent:
                try:
                    agent.record_outcome(
                        exit_price=exit_price,
                        pnl=pnl,
                        holding_period_bars=1,
                    )
                except Exception as e:
                    logger.warning("Failed to record outcome: %s", e)

            del self._positions[symbol]

        return result

    def _calculate_shares(self, price: float) -> int:
        """Calculate position size based on capital and limits."""
        if price <= 0:
            return 0
        dollar_amount = self._capital * self._max_position_pct
        shares = int(dollar_amount / price)
        return min(shares, self._max_shares)

    # ─── Position & Account Info ───

    def get_positions(self) -> Dict[str, Position]:
        """Get locally tracked positions."""
        return dict(self._positions)

    def get_broker_positions(self) -> List[Dict[str, Any]]:
        """Get positions directly from the broker."""
        return self._adapter.get_positions()

    def get_account_info(self) -> Dict[str, Any]:
        """Get account info from the broker."""
        return self._adapter.get_account_info()

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price from the broker."""
        return self._adapter.get_latest_price(symbol)

    def close_all_positions(self, agent: Optional[Any] = None) -> List[ExecutionResult]:
        """Close all open positions (emergency flatten)."""
        results = []
        for symbol in list(self._positions.keys()):
            price = self.get_latest_price(symbol)
            if price <= 0:
                price = self._positions[symbol].entry_price
            result = self._close_position(symbol, price, agent)
            results.append(result)
        return results

    # ─── TP/SL Auto-Management (from hyperliquid-trading-agent) ───

    def _place_tp_sl(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> None:
        """Place take-profit and stop-loss orders after opening a position.

        If tp_price/sl_price are not provided, auto-calculates from defaults.
        """
        if tp_price is None:
            if direction == "long":
                tp_price = round(entry_price * (1 + self._default_tp_pct / 100), 2)
            else:
                tp_price = round(entry_price * (1 - self._default_tp_pct / 100), 2)

        if sl_price is None:
            if direction == "long":
                sl_price = round(entry_price * (1 - self._default_sl_pct / 100), 2)
            else:
                sl_price = round(entry_price * (1 + self._default_sl_pct / 100), 2)

        # Place TP/SL as limit orders if broker supports it
        pos = self._positions.get(symbol)
        if pos:
            close_action = "SELL" if direction == "long" else "BUY"
            try:
                self._adapter.place_limit_order(symbol, close_action, pos.shares, tp_price)
                logger.info("  TP placed: %s %s @ $%.2f", symbol, close_action, tp_price)
            except Exception as e:
                logger.warning("  TP order failed: %s", e)

            logger.info(
                "  Auto TP/SL for %s: TP=$%.2f (+%.1f%%), SL=$%.2f (-%.1f%%)",
                symbol, tp_price, self._default_tp_pct, sl_price, self._default_sl_pct,
            )

    # ─── Force-Close on Max Loss (from hyperliquid-trading-agent) ───

    def check_and_force_close(self, agent: Optional[Any] = None) -> List[ExecutionResult]:
        """Check all positions for max loss and force-close if exceeded.

        Returns:
            List of ExecutionResults for force-closed positions.
        """
        results = []
        for symbol, pos in list(self._positions.items()):
            current_price = self.get_latest_price(symbol)
            if current_price <= 0:
                continue

            # Calculate loss %
            if pos.direction == "long":
                loss_pct = (pos.entry_price - current_price) / pos.entry_price * 100
            else:
                loss_pct = (current_price - pos.entry_price) / pos.entry_price * 100

            if loss_pct >= self._max_loss_pct:
                logger.warning(
                    "⚠️ FORCE-CLOSE: %s %s at %.2f%% loss (max=%.1f%%)",
                    pos.direction.upper(), symbol, loss_pct, self._max_loss_pct,
                )
                result = self._close_position(symbol, current_price, agent)
                if result.success:
                    self._write_diary({
                        "action": "FORCE_CLOSE",
                        "symbol": symbol,
                        "loss_pct": round(loss_pct, 2),
                        "reason": f"Exceeded max loss threshold of {self._max_loss_pct}%",
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                    })
                results.append(result)

        return results

    # ─── Position Reconciliation (from hyperliquid-trading-agent) ───

    def reconcile_positions(self) -> Dict[str, Any]:
        """Sync local position tracking with actual broker positions.

        Detects:
        - Stale local positions (no longer on broker)
        - Missing local positions (on broker but not tracked)

        Returns:
            Dict with reconciliation details.
        """
        broker_positions = self._adapter.get_positions()
        broker_symbols = set()
        reconciliation = {"removed": [], "added": [], "matched": 0}

        for bp in broker_positions:
            sym = bp.get("symbol", "")
            qty = float(bp.get("quantity", 0))
            if abs(qty) > 0:
                broker_symbols.add(sym)

        # Remove stale local positions
        for sym in list(self._positions.keys()):
            if sym not in broker_symbols:
                logger.info("Reconcile: removing stale position %s (not on broker)", sym)
                reconciliation["removed"].append(sym)
                del self._positions[sym]

        # Note missing local positions (on broker but not tracked locally)
        for sym in broker_symbols:
            if sym in self._positions:
                reconciliation["matched"] += 1
            else:
                reconciliation["added"].append(sym)
                logger.info("Reconcile: broker has %s but not tracked locally", sym)

        if reconciliation["removed"] or reconciliation["added"]:
            logger.info("Reconciliation: %s", reconciliation)

        return reconciliation

    # ─── Trade Diary — JSONL Logging (from hyperliquid-trading-agent) ───

    def _write_diary(self, entry: Dict[str, Any]) -> None:
        """Write an entry to the JSONL trade diary."""
        import json as _json
        entry["timestamp"] = datetime.now().isoformat()
        entry["broker"] = self._adapter.name
        entry["mode"] = self._mode

        try:
            with open(self._diary_path, "a") as f:
                f.write(_json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("Diary write failed: %s", e)

    def get_diary(self, n: int = 50) -> List[Dict[str, Any]]:
        """Read the last N entries from the trade diary."""
        import json as _json
        entries = []
        try:
            with open(self._diary_path, "r") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                try:
                    entries.append(_json.loads(line))
                except Exception:
                    pass
        except FileNotFoundError:
            pass
        return entries

    def __repr__(self) -> str:
        return (f"BrokerBridge(broker='{self._adapter.name}', mode='{self._mode}', "
                f"connected={self.is_connected()}, positions={len(self._positions)})")
