"""
Dollar Cost Averaging Bot with Regime-Aware Pausing
=====================================================

Scheduled periodic buys with intelligent pausing when market
conditions are unfavorable. Tracks total invested, current value,
and average cost basis.

Pause conditions (any triggers pause):
- Weekly RSI > 75 (extremely overbought)
- 50-day SMA < 200-day SMA (death cross / bear market)

Usage:
    bot = DCABot(
        connection, order_manager, fetcher,
        symbols=["SPY", "QQQ"],
        dollar_amount=500.0,
    )
    bot.execute_buy_cycle()  # or bot.run(schedule="weekly")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DCASchedule(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class DCAConfig:
    """Configuration for the DCA bot."""

    # Buying parameters
    dollar_amount: float = 500.0
    schedule: DCASchedule = DCASchedule.WEEKLY

    # Target symbols
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    equal_split: bool = True  # split dollar_amount equally among symbols

    # Regime-aware pausing
    enable_regime_pause: bool = True
    rsi_overbought_threshold: float = 75.0
    rsi_length: int = 14
    rsi_timeframe_bars: int = 5  # weekly RSI = 5 daily bars
    sma_fast: int = 50
    sma_slow: int = 200

    # Data
    lookback_duration: str = "1 Y"
    bar_size: str = "1 day"


@dataclass
class DCAPosition:
    """Tracks a DCA position for a single symbol."""
    symbol: str
    total_invested: float = 0.0
    total_shares: float = 0.0
    avg_cost_basis: float = 0.0
    current_price: float = 0.0
    buy_count: int = 0
    first_buy_date: Optional[str] = None
    last_buy_date: Optional[str] = None

    @property
    def current_value(self) -> float:
        return self.total_shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.current_value - self.total_invested

    @property
    def return_pct(self) -> float:
        if self.total_invested <= 0:
            return 0.0
        return (self.current_value / self.total_invested - 1) * 100


class DCABot:
    """Dollar Cost Averaging bot with regime awareness.

    Performs scheduled buys of target symbols, with intelligent
    pausing when market is overbought or in a bear trend.

    Args:
        connection: An IBInsyncConnection instance.
        order_manager: OrderManager for trade execution.
        fetcher: HistoricalDataFetcher for bar data.
        risk_manager: Optional RiskManager for risk controls.
        notifier: Optional AlertDispatcher for notifications.
        symbols: Override symbols list.
        dollar_amount: Override dollar amount per cycle.
        config: DCAConfig with all parameters.
    """

    def __init__(
        self,
        connection: Any,
        order_manager: Any,
        fetcher: Any,
        risk_manager: Any = None,
        notifier: Any = None,
        symbols: Optional[List[str]] = None,
        dollar_amount: Optional[float] = None,
        config: Optional[DCAConfig] = None,
    ) -> None:
        self.connection = connection
        self.order_manager = order_manager
        self.fetcher = fetcher
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.config = config or DCAConfig()

        if symbols:
            self.config.symbols = symbols
        if dollar_amount:
            self.config.dollar_amount = dollar_amount

        self._positions: Dict[str, DCAPosition] = {
            sym: DCAPosition(symbol=sym) for sym in self.config.symbols
        }
        self._buy_history: List[dict] = []
        self._paused_cycles: int = 0
        self._running = False

    # ─── Regime Checks ───

    def _check_regime(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check if market conditions warrant pausing.

        Args:
            df: Daily OHLCV DataFrame (at least 200 bars).

        Returns:
            Tuple of (should_pause, reason).
        """
        if not self.config.enable_regime_pause:
            return False, "Regime check disabled"

        close = df["close"]
        reasons = []

        # Weekly RSI check (using daily close with weekly period)
        weekly_close = close.resample("W").last().dropna() if hasattr(close.index, "freq") else close
        rsi = self._calculate_rsi(close, self.config.rsi_length)

        if len(rsi) > 0:
            current_rsi = rsi.iloc[-1]
            if current_rsi > self.config.rsi_overbought_threshold:
                reasons.append(f"RSI={current_rsi:.1f} > {self.config.rsi_overbought_threshold}")

        # Death cross check (50 SMA < 200 SMA)
        if len(close) >= self.config.sma_slow:
            sma_fast = close.rolling(window=self.config.sma_fast).mean().iloc[-1]
            sma_slow = close.rolling(window=self.config.sma_slow).mean().iloc[-1]

            if sma_fast < sma_slow:
                reasons.append(f"Death cross: SMA{self.config.sma_fast}={sma_fast:.2f} < SMA{self.config.sma_slow}={sma_slow:.2f}")

        if reasons:
            reason = " | ".join(reasons)
            logger.warning("DCA PAUSED: %s", reason)
            return True, reason

        return False, "Conditions normal"

    @staticmethod
    def _calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Calculate RSI from a price series."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ─── Buy Execution ───

    def execute_buy_cycle(self) -> List[dict]:
        """Execute one DCA buy cycle for all symbols.

        Checks regime conditions first. If paused, no buys are made.

        Returns:
            List of buy records with execution details.
        """
        cfg = self.config
        results: List[dict] = []

        # Fetch data for regime check (use first symbol as market proxy)
        proxy_symbol = cfg.symbols[0]
        try:
            df = self.fetcher.fetch_bars(
                proxy_symbol,
                duration=cfg.lookback_duration,
                bar_size=cfg.bar_size,
            )

            if not df.empty:
                should_pause, reason = self._check_regime(df)
                if should_pause:
                    self._paused_cycles += 1
                    msg = f"DCA paused (cycle #{self._paused_cycles}): {reason}"
                    logger.info(msg)
                    if self.notifier:
                        self.notifier.dispatch("DCA Paused", msg, priority="WARNING")
                    return []
        except Exception as e:
            logger.warning("Regime check failed: %s. Proceeding with buys.", e)

        # Calculate per-symbol amount
        n_symbols = len(cfg.symbols)
        per_symbol_amount = cfg.dollar_amount / n_symbols if cfg.equal_split else cfg.dollar_amount

        for symbol in cfg.symbols:
            try:
                result = self._buy_symbol(symbol, per_symbol_amount)
                results.append(result)
            except Exception as e:
                logger.error("Failed to buy %s: %s", symbol, e)
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

        self._buy_history.extend(results)

        # Summary notification
        if self.notifier and results:
            bought = [r for r in results if r.get("status") == "filled"]
            total_spent = sum(r.get("total_cost", 0) for r in bought)
            self.notifier.dispatch(
                "DCA Buy Cycle",
                f"Bought {len(bought)}/{len(cfg.symbols)} symbols. "
                f"Total: ${total_spent:,.2f}",
            )

        return results

    def _buy_symbol(self, symbol: str, dollar_amount: float) -> dict:
        """Buy a single symbol with a fixed dollar amount.

        Args:
            symbol: Ticker symbol.
            dollar_amount: Dollar amount to invest.

        Returns:
            Dict with buy details.
        """
        # Get current price
        df = self.fetcher.fetch_bars(symbol, duration="5 D", bar_size="1 day")
        if df.empty:
            raise ValueError(f"No price data for {symbol}")

        current_price = df["close"].iloc[-1]
        quantity = max(1, int(dollar_amount / current_price))
        total_cost = quantity * current_price

        # Risk manager check
        if self.risk_manager and not self.risk_manager.can_trade():
            logger.warning("RiskManager blocked DCA buy for %s", symbol)
            return {
                "symbol": symbol,
                "status": "blocked",
                "reason": "risk_manager",
                "timestamp": datetime.now().isoformat(),
            }

        # Place market order
        trade = self.order_manager.market_order(
            symbol=symbol,
            action="BUY",
            quantity=quantity,
        )

        # Update position tracking
        pos = self._positions[symbol]
        pos.total_invested += total_cost
        pos.total_shares += quantity
        pos.avg_cost_basis = pos.total_invested / pos.total_shares if pos.total_shares > 0 else 0
        pos.current_price = current_price
        pos.buy_count += 1
        if pos.first_buy_date is None:
            pos.first_buy_date = date.today().isoformat()
        pos.last_buy_date = date.today().isoformat()

        logger.info(
            "DCA Buy: %d shares of %s @ $%.2f ($%.2f total) | "
            "Avg cost: $%.2f | Total invested: $%.2f | Buy #%d",
            quantity, symbol, current_price, total_cost,
            pos.avg_cost_basis, pos.total_invested, pos.buy_count,
        )

        return {
            "symbol": symbol,
            "status": "filled",
            "quantity": quantity,
            "price": current_price,
            "total_cost": total_cost,
            "avg_cost_basis": pos.avg_cost_basis,
            "total_invested": pos.total_invested,
            "total_shares": pos.total_shares,
            "buy_number": pos.buy_count,
            "timestamp": datetime.now().isoformat(),
        }

    # ─── Scheduled Loop ───

    def run(
        self,
        schedule: Optional[str] = None,
        max_cycles: Optional[int] = None,
    ) -> None:
        """Run the DCA bot on a schedule.

        Args:
            schedule: Override schedule ("daily", "weekly", "monthly").
            max_cycles: Stop after N cycles (None = run forever).
        """
        if schedule:
            self.config.schedule = DCASchedule(schedule)

        interval_map = {
            DCASchedule.DAILY: 86400,
            DCASchedule.WEEKLY: 604800,
            DCASchedule.MONTHLY: 2592000,
        }
        interval = interval_map[self.config.schedule]

        self._running = True
        cycle = 0

        logger.info(
            "DCA Bot started: %s schedule, $%.2f per cycle, symbols=%s",
            self.config.schedule.value,
            self.config.dollar_amount,
            self.config.symbols,
        )

        try:
            while self._running:
                if max_cycles and cycle >= max_cycles:
                    logger.info("Max cycles reached (%d). Stopping.", max_cycles)
                    break

                self.execute_buy_cycle()
                cycle += 1

                logger.info("Cycle %d complete. Next in %ds.", cycle, interval)
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("DCA Bot stopped by user.")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the bot to stop."""
        self._running = False
        logger.info("DCA Bot stop requested.")

    # ─── Status ───

    def get_portfolio_summary(self) -> dict:
        """Get summary of all DCA positions.

        Returns:
            Dict with per-symbol and aggregate metrics.
        """
        positions = {}
        total_invested = 0.0
        total_value = 0.0

        for symbol, pos in self._positions.items():
            positions[symbol] = {
                "total_invested": round(pos.total_invested, 2),
                "total_shares": pos.total_shares,
                "avg_cost_basis": round(pos.avg_cost_basis, 2),
                "current_price": round(pos.current_price, 2),
                "current_value": round(pos.current_value, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "return_pct": round(pos.return_pct, 2),
                "buy_count": pos.buy_count,
            }
            total_invested += pos.total_invested
            total_value += pos.current_value

        return {
            "positions": positions,
            "total_invested": round(total_invested, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(total_value - total_invested, 2),
            "total_return_pct": round(
                (total_value / total_invested - 1) * 100 if total_invested > 0 else 0, 2
            ),
            "total_buys": sum(p.buy_count for p in self._positions.values()),
            "paused_cycles": self._paused_cycles,
            "schedule": self.config.schedule.value,
        }

    def get_buy_history(self) -> List[dict]:
        """Return all buy records."""
        return list(self._buy_history)
