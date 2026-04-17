"""
Pairs Trading Bot — Statistical Arbitrage Strategy
====================================================

Implements a pairs trading strategy using cointegration analysis,
z-score signals, and dollar-neutral position sizing.

Usage:
    bot = PairsTradingBot(connection, order_manager, notifier=dispatcher)
    if bot.test_cointegration("AAPL", "MSFT", lookback=252):
        bot.run("AAPL", "MSFT", entry_z=2.0, exit_z=0.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PairState(Enum):
    """Current state of the pairs trade."""
    FLAT = "flat"
    LONG_SPREAD = "long_spread"   # long leg1, short leg2
    SHORT_SPREAD = "short_spread"  # short leg1, long leg2


@dataclass
class PairSignal:
    """Signal output from pairs analysis."""
    timestamp: datetime
    symbol_a: str
    symbol_b: str
    spread: float
    z_score: float
    hedge_ratio: float
    state: PairState
    action: str = "HOLD"


@dataclass
class PairsTrade:
    """Record of a pairs trade entry/exit."""
    entry_time: datetime
    symbol_a: str
    symbol_b: str
    direction: PairState
    qty_a: int
    qty_b: int
    entry_price_a: float
    entry_price_b: float
    entry_z: float
    hedge_ratio: float
    exit_time: Optional[datetime] = None
    exit_price_a: float = 0.0
    exit_price_b: float = 0.0
    exit_z: float = 0.0
    pnl: float = 0.0
    closed: bool = False


class PairsTradingBot:
    """Statistical arbitrage pairs trading bot.

    Uses the Engle-Granger cointegration test to identify
    mean-reverting pairs, then trades z-score deviations.

    Args:
        connection: An IBInsyncConnection instance.
        order_manager: OrderManager for trade execution.
        notifier: Optional AlertDispatcher for notifications.
        capital: Capital allocated to the strategy.
    """

    def __init__(
        self,
        connection: Any,
        order_manager: Any,
        notifier: Any = None,
        capital: float = 100000.0,
    ) -> None:
        self.connection = connection
        self.order_manager = order_manager
        self.notifier = notifier
        self.capital = capital

        self._state = PairState.FLAT
        self._current_trade: Optional[PairsTrade] = None
        self._trade_history: list[PairsTrade] = []
        self._hedge_ratio: float = 1.0

    def test_cointegration(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: Optional[pd.Series] = None,
        prices_b: Optional[pd.Series] = None,
        significance: float = 0.05,
    ) -> Tuple[bool, float, float]:
        """Test for cointegration between two price series using Engle-Granger.

        Args:
            symbol_a: First symbol.
            symbol_b: Second symbol.
            prices_a: Price series for symbol_a (if None, will be fetched).
            prices_b: Price series for symbol_b (if None, will be fetched).
            significance: P-value threshold for cointegration.

        Returns:
            Tuple of (is_cointegrated, p_value, test_statistic).
        """
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            raise ImportError(
                "statsmodels is required for cointegration testing. "
                "Install with: pip install statsmodels"
            )

        if prices_a is None or prices_b is None:
            raise ValueError(
                "Price series must be provided. Use HistoricalDataFetcher "
                "to obtain prices before calling test_cointegration."
            )

        prices_a = prices_a.dropna()
        prices_b = prices_b.dropna()
        common_idx = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.loc[common_idx]
        prices_b = prices_b.loc[common_idx]

        if len(prices_a) < 30:
            logger.warning(
                "Insufficient data for cointegration test: %d points",
                len(prices_a),
            )
            return False, 1.0, 0.0

        test_stat, p_value, critical_values = coint(prices_a, prices_b)
        is_cointegrated = p_value < significance

        logger.info(
            "Cointegration test %s/%s: stat=%.4f, p=%.4f, "
            "cointegrated=%s (threshold=%.2f)",
            symbol_a, symbol_b, test_stat, p_value,
            is_cointegrated, significance,
        )

        return is_cointegrated, p_value, test_stat

    def calculate_hedge_ratio(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> float:
        """Calculate the hedge ratio via OLS regression.

        Regresses prices_a on prices_b to find the optimal
        number of shares of B per share of A.

        Args:
            prices_a: Dependent variable price series.
            prices_b: Independent variable price series.

        Returns:
            The hedge ratio (beta coefficient).
        """
        try:
            from statsmodels.api import OLS, add_constant
        except ImportError:
            raise ImportError(
                "statsmodels is required. Install with: pip install statsmodels"
            )

        common_idx = prices_a.index.intersection(prices_b.index)
        y = prices_a.loc[common_idx].values
        X = add_constant(prices_b.loc[common_idx].values)

        model = OLS(y, X).fit()
        hedge_ratio = model.params[1]
        r_squared = model.rsquared

        logger.info(
            "Hedge ratio: %.4f (R²=%.4f)", hedge_ratio, r_squared,
        )

        self._hedge_ratio = hedge_ratio
        return hedge_ratio

    def calculate_spread(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: Optional[float] = None,
    ) -> pd.Series:
        """Calculate the spread between two price series.

        spread = prices_a - hedge_ratio * prices_b

        Args:
            prices_a: First price series.
            prices_b: Second price series.
            hedge_ratio: Hedge ratio (uses stored value if None).

        Returns:
            The spread as a pandas Series.
        """
        if hedge_ratio is None:
            hedge_ratio = self._hedge_ratio

        common_idx = prices_a.index.intersection(prices_b.index)
        spread = prices_a.loc[common_idx] - hedge_ratio * prices_b.loc[common_idx]
        return spread

    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: int = 20,
    ) -> pd.Series:
        """Calculate the z-score of the spread.

        z = (spread - rolling_mean) / rolling_std

        Args:
            spread: The spread series.
            lookback: Rolling window for mean/std calculation.

        Returns:
            Z-score series.
        """
        mean = spread.rolling(window=lookback).mean()
        std = spread.rolling(window=lookback).std()
        zscore = (spread - mean) / std
        return zscore

    def generate_signal(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: pd.Series,
        prices_b: pd.Series,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 20,
    ) -> PairSignal:
        """Generate a trading signal based on current z-score.

        Entry at |z| > entry_z, exit at |z| < exit_z.

        Args:
            symbol_a: First symbol.
            symbol_b: Second symbol.
            prices_a: First price series.
            prices_b: Second price series.
            entry_z: Z-score threshold for entry.
            exit_z: Z-score threshold for exit.
            lookback: Rolling lookback window.

        Returns:
            PairSignal with current state and recommended action.
        """
        spread = self.calculate_spread(prices_a, prices_b)
        zscore = self.calculate_zscore(spread, lookback)

        current_z = zscore.iloc[-1]
        current_spread = spread.iloc[-1]

        action = "HOLD"

        if self._state == PairState.FLAT:
            if current_z > entry_z:
                action = "SHORT_SPREAD"
            elif current_z < -entry_z:
                action = "LONG_SPREAD"
        elif self._state == PairState.LONG_SPREAD:
            if current_z > -exit_z:
                action = "EXIT"
            if current_z > entry_z:
                action = "REVERSE_TO_SHORT"
        elif self._state == PairState.SHORT_SPREAD:
            if current_z < exit_z:
                action = "EXIT"
            if current_z < -entry_z:
                action = "REVERSE_TO_LONG"

        signal = PairSignal(
            timestamp=datetime.now(),
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            spread=current_spread,
            z_score=current_z,
            hedge_ratio=self._hedge_ratio,
            state=self._state,
            action=action,
        )

        logger.info(
            "Signal: %s/%s z=%.2f spread=%.4f action=%s state=%s",
            symbol_a, symbol_b, current_z, current_spread,
            action, self._state.value,
        )
        return signal

    def _calculate_position_sizes(
        self,
        price_a: float,
        price_b: float,
    ) -> Tuple[int, int]:
        """Calculate dollar-neutral position sizes.

        Allocates half the capital to each leg, adjusting for
        the hedge ratio.

        Args:
            price_a: Current price of symbol A.
            price_b: Current price of symbol B.

        Returns:
            Tuple of (quantity_a, quantity_b).
        """
        half_capital = self.capital / 2.0
        qty_a = int(half_capital / price_a)
        qty_b = int((half_capital / price_b) * abs(self._hedge_ratio))

        value_a = qty_a * price_a
        value_b = qty_b * price_b
        logger.info(
            "Position sizes: %d x $%.2f = $%.0f | %d x $%.2f = $%.0f",
            qty_a, price_a, value_a, qty_b, price_b, value_b,
        )
        return qty_a, qty_b

    def enter_trade(
        self,
        symbol_a: str,
        symbol_b: str,
        direction: PairState,
        price_a: float,
        price_b: float,
        z_score: float,
    ) -> PairsTrade:
        """Enter a pairs trade (long spread or short spread).

        Long spread: BUY A, SELL B
        Short spread: SELL A, BUY B

        Args:
            symbol_a: First symbol.
            symbol_b: Second symbol.
            direction: LONG_SPREAD or SHORT_SPREAD.
            price_a: Current price of A.
            price_b: Current price of B.
            z_score: Current z-score at entry.

        Returns:
            PairsTrade record.
        """
        qty_a, qty_b = self._calculate_position_sizes(price_a, price_b)

        if direction == PairState.LONG_SPREAD:
            self.order_manager.market_order(symbol_a, "BUY", qty_a)
            self.order_manager.market_order(symbol_b, "SELL", qty_b)
        elif direction == PairState.SHORT_SPREAD:
            self.order_manager.market_order(symbol_a, "SELL", qty_a)
            self.order_manager.market_order(symbol_b, "BUY", qty_b)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        trade = PairsTrade(
            entry_time=datetime.now(),
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            direction=direction,
            qty_a=qty_a,
            qty_b=qty_b,
            entry_price_a=price_a,
            entry_price_b=price_b,
            entry_z=z_score,
            hedge_ratio=self._hedge_ratio,
        )

        self._current_trade = trade
        self._state = direction

        msg = (
            f"Pairs entry: {direction.value} "
            f"{qty_a} {symbol_a} @ ${price_a:.2f} / "
            f"{qty_b} {symbol_b} @ ${price_b:.2f} (z={z_score:.2f})"
        )
        logger.info(msg)
        if self.notifier:
            self.notifier.info(msg)

        return trade

    def exit_trade(
        self,
        price_a: float,
        price_b: float,
        z_score: float,
    ) -> Optional[PairsTrade]:
        """Exit the current pairs trade.

        Args:
            price_a: Current price of A.
            price_b: Current price of B.
            z_score: Current z-score at exit.

        Returns:
            The closed PairsTrade with P&L, or None if no position.
        """
        if self._current_trade is None or self._state == PairState.FLAT:
            logger.warning("No active pairs trade to exit")
            return None

        trade = self._current_trade
        if trade.direction == PairState.LONG_SPREAD:
            self.order_manager.market_order(trade.symbol_a, "SELL", trade.qty_a)
            self.order_manager.market_order(trade.symbol_b, "BUY", trade.qty_b)
            pnl_a = (price_a - trade.entry_price_a) * trade.qty_a
            pnl_b = (trade.entry_price_b - price_b) * trade.qty_b
        else:
            self.order_manager.market_order(trade.symbol_a, "BUY", trade.qty_a)
            self.order_manager.market_order(trade.symbol_b, "SELL", trade.qty_b)
            pnl_a = (trade.entry_price_a - price_a) * trade.qty_a
            pnl_b = (price_b - trade.entry_price_b) * trade.qty_b

        trade.exit_time = datetime.now()
        trade.exit_price_a = price_a
        trade.exit_price_b = price_b
        trade.exit_z = z_score
        trade.pnl = pnl_a + pnl_b
        trade.closed = True

        self._trade_history.append(trade)
        self._current_trade = None
        self._state = PairState.FLAT

        msg = (
            f"Pairs exit: P&L=${trade.pnl:+.2f} "
            f"(A: ${pnl_a:+.2f}, B: ${pnl_b:+.2f}) z={z_score:.2f}"
        )
        logger.info(msg)
        if self.notifier:
            level = "info" if trade.pnl >= 0 else "warning"
            if hasattr(self.notifier, level):
                getattr(self.notifier, level)(msg)

        return trade

    def get_trade_history(self) -> list[PairsTrade]:
        """Return all historical trades."""
        return list(self._trade_history)

    def get_performance_summary(self) -> dict:
        """Calculate strategy performance metrics.

        Returns:
            Dictionary with total_trades, winning_trades, win_rate,
            total_pnl, avg_pnl, max_win, max_loss.
        """
        if not self._trade_history:
            return {"total_trades": 0}

        pnls = [t.pnl for t in self._trade_history if t.closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls) if pnls else 0.0,
            "max_win": max(pnls) if pnls else 0.0,
            "max_loss": min(pnls) if pnls else 0.0,
            "profit_factor": (
                sum(wins) / abs(sum(losses))
                if losses and sum(losses) != 0
                else float("inf")
            ),
        }
