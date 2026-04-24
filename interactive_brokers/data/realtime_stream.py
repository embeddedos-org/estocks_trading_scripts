"""
Real-Time Market Data Streaming for Interactive Brokers
========================================================

Provides thread-safe real-time tick data streaming and bar aggregation
from IB Gateway/TWS.

Usage:
    stream = RealtimeDataStream(connection)
    stream.subscribe("AAPL", callback=on_tick)
    stream.subscribe("MSFT", callback=on_tick)
    snapshot = stream.get_snapshot("AAPL")
    stream.unsubscribe("AAPL")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Represents a single market data tick."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    last_size: int = 0
    volume: int = 0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    vwap: float = 0.0

    @property
    def mid(self) -> float:
        """Calculate mid price from bid/ask."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid
        if mid > 0:
            return (self.spread / mid) * 100.0
        return 0.0


@dataclass
class AggregatedBar:
    """Represents an aggregated OHLCV bar built from ticks."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int
    vwap: float = 0.0
    duration_seconds: int = 60


class BarAggregator:
    """Aggregates raw ticks into custom-timeframe OHLCV bars.

    Args:
        symbol: Ticker symbol.
        interval_seconds: Bar duration in seconds (e.g., 60 for 1-min bars).
        callback: Function called with each completed AggregatedBar.
    """

    def __init__(
        self,
        symbol: str,
        interval_seconds: int = 60,
        callback: Optional[Callable[[AggregatedBar], None]] = None,
    ) -> None:
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.callback = callback

        self._current_bar: Optional[AggregatedBar] = None
        self._bar_start_time: Optional[datetime] = None
        self._volume_price_sum: float = 0.0
        self._total_volume: int = 0
        self._completed_bars: List[AggregatedBar] = []
        self._lock = threading.Lock()

    def on_tick(self, tick: TickData) -> Optional[AggregatedBar]:
        """Process an incoming tick and aggregate into bars.

        Args:
            tick: The incoming TickData.

        Returns:
            A completed AggregatedBar if the interval has elapsed, else None.
        """
        with self._lock:
            price = tick.last if tick.last > 0 else tick.mid
            if price <= 0:
                return None

            now = tick.timestamp
            completed_bar = None

            if self._current_bar is None or self._should_close_bar(now):
                if self._current_bar is not None:
                    self._current_bar.close = price
                    if self._total_volume > 0:
                        self._current_bar.vwap = (
                            self._volume_price_sum / self._total_volume
                        )
                    completed_bar = self._current_bar
                    self._completed_bars.append(completed_bar)

                    if self.callback:
                        try:
                            self.callback(completed_bar)
                        except Exception as e:
                            logger.error("Bar callback error: %s", e)

                self._current_bar = AggregatedBar(
                    symbol=self.symbol,
                    timestamp=now,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0,
                    tick_count=0,
                    duration_seconds=self.interval_seconds,
                )
                self._bar_start_time = now
                self._volume_price_sum = 0.0
                self._total_volume = 0

            bar = self._current_bar
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
            bar.volume += tick.last_size
            bar.tick_count += 1
            self._volume_price_sum += price * tick.last_size
            self._total_volume += tick.last_size

            return completed_bar

    def _should_close_bar(self, now: datetime) -> bool:
        """Check if the current bar interval has elapsed."""
        if self._bar_start_time is None:
            return True
        elapsed = (now - self._bar_start_time).total_seconds()
        return elapsed >= self.interval_seconds

    def get_completed_bars(self) -> List[AggregatedBar]:
        """Return all completed bars."""
        with self._lock:
            return list(self._completed_bars)

    def reset(self) -> None:
        """Clear all bar state."""
        with self._lock:
            self._current_bar = None
            self._bar_start_time = None
            self._completed_bars.clear()
            self._volume_price_sum = 0.0
            self._total_volume = 0


class RealtimeDataStream:
    """Manages real-time market data subscriptions via IB.

    Thread-safe streaming with support for multiple concurrent
    subscriptions, tick callbacks, and bar aggregation.

    Args:
        connection: An IBInsyncConnection instance.
    """

    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self._subscriptions: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._snapshots: Dict[str, TickData] = {}
        self._aggregators: Dict[str, BarAggregator] = {}
        self._lock = threading.Lock()
        self._running = False

    def _create_contract(
        self,
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Any:
        """Create and qualify a contract for streaming."""
        try:
            from ib_insync import Stock, Contract

            if sec_type == "STK":
                contract = Stock(symbol, exchange, currency)
            else:
                contract = Contract(
                    symbol=symbol,
                    secType=sec_type,
                    exchange=exchange,
                    currency=currency,
                )
            self.connection.qualifyContracts(contract)
            return contract
        except ImportError:
            raise ImportError(
                "ib_insync is required for RealtimeDataStream. "
                "Install with: pip install ib_insync"
            )

    def _on_pending_tickers(self, tickers: List[Any]) -> None:
        """Internal callback for ib_insync pending ticker updates."""
        for ticker in tickers:
            symbol = ticker.contract.symbol
            tick = TickData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=ticker.bid if ticker.bid == ticker.bid else 0.0,
                ask=ticker.ask if ticker.ask == ticker.ask else 0.0,
                last=ticker.last if ticker.last == ticker.last else 0.0,
                bid_size=int(ticker.bidSize) if ticker.bidSize == ticker.bidSize else 0,
                ask_size=int(ticker.askSize) if ticker.askSize == ticker.askSize else 0,
                last_size=int(ticker.lastSize) if ticker.lastSize == ticker.lastSize else 0,
                volume=int(ticker.volume) if ticker.volume == ticker.volume else 0,
                high=ticker.high if ticker.high == ticker.high else 0.0,
                low=ticker.low if ticker.low == ticker.low else 0.0,
                open=ticker.open if ticker.open == ticker.open else 0.0,
                close=ticker.close if ticker.close == ticker.close else 0.0,
            )

            with self._lock:
                self._snapshots[symbol] = tick

                if symbol in self._aggregators:
                    self._aggregators[symbol].on_tick(tick)

            for callback in self._callbacks.get(symbol, []):
                try:
                    callback(tick)
                except Exception as e:
                    logger.error("Tick callback error for %s: %s", symbol, e)

    def subscribe(
        self,
        symbol: str,
        callback: Optional[Callable[[TickData], None]] = None,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        bar_interval: Optional[int] = None,
        bar_callback: Optional[Callable[[AggregatedBar], None]] = None,
    ) -> None:
        """Subscribe to real-time data for a symbol.

        Args:
            symbol: Ticker symbol.
            callback: Function called on each tick update.
            sec_type: Security type.
            exchange: Exchange.
            currency: Currency.
            bar_interval: If set, aggregate ticks into bars of this many seconds.
            bar_callback: Function called on each completed aggregated bar.
        """
        with self._lock:
            if symbol in self._subscriptions:
                logger.warning("Already subscribed to %s", symbol)
                if callback:
                    self._callbacks[symbol].append(callback)
                return

        contract = self._create_contract(symbol, sec_type, exchange, currency)

        ticker = self.connection.reqMktData(contract)
        logger.info("Subscribed to real-time data for %s", symbol)

        with self._lock:
            self._subscriptions[symbol] = {
                "contract": contract,
                "ticker": ticker,
            }
            if callback:
                self._callbacks[symbol].append(callback)

            if bar_interval:
                self._aggregators[symbol] = BarAggregator(
                    symbol=symbol,
                    interval_seconds=bar_interval,
                    callback=bar_callback,
                )

        if not self._running:
            self._running = True
            if hasattr(self.connection, 'ib'):
                self.connection.ib.pendingTickersEvent += self._on_pending_tickers

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from real-time data for a symbol.

        Args:
            symbol: Ticker symbol to unsubscribe.
        """
        with self._lock:
            sub = self._subscriptions.pop(symbol, None)
            self._callbacks.pop(symbol, None)
            self._aggregators.pop(symbol, None)
            self._snapshots.pop(symbol, None)

        if sub and hasattr(self.connection, 'ib'):
            self.connection.ib.cancelMktData(sub["contract"])
            logger.info("Unsubscribed from %s", symbol)
        else:
            logger.warning("No active subscription for %s", symbol)

    def subscribe_multiple(
        self,
        symbols: List[str],
        callback: Optional[Callable[[TickData], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Subscribe to multiple symbols at once.

        Args:
            symbols: List of ticker symbols.
            callback: Shared callback for all symbols.
        """
        for symbol in symbols:
            self.subscribe(symbol, callback=callback, **kwargs)

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all active subscriptions."""
        symbols = list(self._subscriptions.keys())
        for symbol in symbols:
            self.unsubscribe(symbol)
        self._running = False
        logger.info("Unsubscribed from all symbols")

    def get_snapshot(self, symbol: str) -> Optional[TickData]:
        """Get the latest tick snapshot for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Latest TickData if available, None otherwise.
        """
        with self._lock:
            return self._snapshots.get(symbol)

    def get_all_snapshots(self) -> Dict[str, TickData]:
        """Get snapshots for all subscribed symbols.

        Returns:
            Dictionary mapping symbol → latest TickData.
        """
        with self._lock:
            return dict(self._snapshots)

    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        with self._lock:
            return list(self._subscriptions.keys())

    def is_subscribed(self, symbol: str) -> bool:
        """Check if a symbol is currently subscribed."""
        with self._lock:
            return symbol in self._subscriptions
