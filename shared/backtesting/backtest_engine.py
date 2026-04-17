"""Backtesting engine for evaluating trading strategies on historical data.

Supports arbitrary strategy callables, computes standard risk/return
metrics, and produces an equity curve with a full trade log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest performance metrics and logs."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)


class BacktestEngine:
    """Simple event-driven backtester.

    Args:
        initial_capital: Starting portfolio value in dollars.
        commission: Commission rate as a fraction of trade value (e.g. 0.001 = 0.1%).
    """

    ANNUALIZATION_FACTOR = 252  # trading days per year

    def __init__(
        self, initial_capital: float = 100_000.0, commission: float = 0.001
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self._data: pd.DataFrame | None = None
        self._equity_curve: list[float] = []
        self._trade_log: list[dict[str, Any]] = []
        self._daily_returns: list[float] = []

    def load_data(self, df: pd.DataFrame) -> None:
        """Load OHLCV data for backtesting.

        The DataFrame must contain columns: date (or datetime), open, high,
        low, close, volume.  Column names are normalised to lowercase.

        Args:
            df: A pandas DataFrame with OHLCV price data.

        Raises:
            ValueError: If required columns are missing.
        """
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        if "datetime" in df.columns and "date" not in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        self._data = df

    def run(self, strategy_fn: Callable[[int, pd.Series, int, float], int]) -> BacktestResult:
        """Execute a backtest using the provided strategy function.

        The strategy callable receives ``(index, row, position, capital)``
        and must return a signal:

        * ``1``  — buy / go long
        * ``-1`` — sell / go short (or close long)
        * ``0``  — hold / do nothing

        Args:
            strategy_fn: Strategy callable producing trade signals.

        Returns:
            A ``BacktestResult`` with computed metrics, equity curve, and
            trade log.

        Raises:
            RuntimeError: If no data has been loaded via ``load_data``.
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        capital = self.initial_capital
        position = 0  # +1 = long, 0 = flat
        entry_price = 0.0
        shares = 0

        self._equity_curve = []
        self._trade_log = []
        self._daily_returns = []

        prev_equity = capital

        for idx, row in self._data.iterrows():
            price = float(row["close"])
            signal = strategy_fn(int(idx), row, position, capital)

            if signal == 1 and position == 0:
                shares = int(capital // price)
                if shares <= 0:
                    equity = capital
                else:
                    cost = shares * price
                    commission_paid = cost * self.commission
                    capital -= cost + commission_paid
                    entry_price = price
                    position = 1
                    self._trade_log.append(
                        {
                            "type": "BUY",
                            "date": str(row["date"]),
                            "price": price,
                            "shares": shares,
                            "commission": commission_paid,
                        }
                    )
                equity = capital + shares * price

            elif signal == -1 and position == 1:
                proceeds = shares * price
                commission_paid = proceeds * self.commission
                capital += proceeds - commission_paid
                pnl = (price - entry_price) * shares - commission_paid
                self._trade_log.append(
                    {
                        "type": "SELL",
                        "date": str(row["date"]),
                        "price": price,
                        "shares": shares,
                        "commission": commission_paid,
                        "pnl": pnl,
                    }
                )
                shares = 0
                position = 0
                entry_price = 0.0
                equity = capital

            else:
                equity = capital + shares * price

            self._equity_curve.append(equity)

            if prev_equity != 0:
                daily_ret = (equity - prev_equity) / prev_equity
            else:
                daily_ret = 0.0
            self._daily_returns.append(daily_ret)
            prev_equity = equity

        return self._compute_metrics()

    def _compute_metrics(self) -> BacktestResult:
        """Compute risk/return metrics from the completed backtest."""
        equity = self._equity_curve
        if not equity:
            return BacktestResult()

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        returns = np.array(self._daily_returns, dtype=np.float64)

        # Sharpe ratio (annualized)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = (
            (mean_ret / std_ret) * math.sqrt(self.ANNUALIZATION_FACTOR)
            if std_ret > 0
            else 0.0
        )

        # Sortino ratio (annualized, using downside deviation)
        downside = returns[returns < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = (
            (mean_ret / downside_std) * math.sqrt(self.ANNUALIZATION_FACTOR)
            if downside_std > 0
            else 0.0
        )

        # Max drawdown
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Trade-level metrics
        sell_trades = [t for t in self._trade_log if t["type"] == "SELL"]
        total_trades = len(sell_trades)
        winners = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losers = [t for t in sell_trades if t.get("pnl", 0) <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in winners) if winners else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        return BacktestResult(
            total_return=round(total_return, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 6),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            total_trades=total_trades,
            equity_curve=equity,
            trade_log=self._trade_log,
        )
