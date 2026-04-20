"""
Portfolio Backtesting Engine
==============================

High-level API for running portfolio-level backtests using bt.

Usage:
    from portfolio_backtester.portfolio_engine import PortfolioEngine, PortfolioBacktestConfig

    engine = PortfolioEngine(PortfolioBacktestConfig(rebalance_freq="monthly"))
    result = engine.run("momentum", price_data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import bt as bt_lib  # type: ignore[import-untyped]
    _HAS_BT = True
except ImportError:
    _HAS_BT = False

try:
    from shared.backtesting.backtest_engine_v2 import BacktestResultV2
    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False


@dataclass
class PortfolioBacktestConfig:
    """Configuration for portfolio backtests.

    Attributes:
        rebalance_freq: "daily", "weekly", "monthly", "quarterly"
        initial_capital: Starting portfolio value
        commission: Commission per trade (fraction)
        benchmark: Optional benchmark ticker column name
    """
    rebalance_freq: str = "monthly"
    initial_capital: float = 100_000.0
    commission: float = 0.001
    benchmark: Optional[str] = None


class PortfolioEngine:
    """Run portfolio-level backtests with various allocation strategies."""

    STRATEGY_MAP = {
        "equal_weight": "EqualWeightAlgo",
        "momentum": "MomentumAlgo",
        "risk_parity": "RiskParityAlgo",
        "mean_variance": "MeanVarianceAlgo",
        "tactical": "TacticalAllocationAlgo",
    }

    def __init__(self, config: Optional[PortfolioBacktestConfig] = None):
        if not _HAS_BT:
            raise ImportError("bt is required. Install: pip install bt")
        self.config = config or PortfolioBacktestConfig()

    def run(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        **strategy_kwargs,
    ) -> Any:
        """Run a portfolio backtest.

        Args:
            strategy_name: One of "equal_weight", "momentum", "risk_parity",
                "mean_variance", "tactical"
            data: DataFrame of prices (columns = tickers, index = dates)
            **strategy_kwargs: Additional kwargs for the strategy algo

        Returns:
            BacktestResultV2 if available, otherwise bt Result object
        """
        if strategy_name not in self.STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(self.STRATEGY_MAP.keys())}"
            )

        from portfolio_backtester import strategies as strat_mod

        algo_cls = getattr(strat_mod, self.STRATEGY_MAP[strategy_name])
        if algo_cls is None:
            raise ImportError(f"Strategy {strategy_name} requires bt library")

        # Build rebalance algo
        freq_map = {
            "daily": bt_lib.algos.RunDaily,
            "weekly": bt_lib.algos.RunWeekly,
            "monthly": bt_lib.algos.RunMonthly,
            "quarterly": bt_lib.algos.RunQuarterly,
        }
        rebal_cls = freq_map.get(self.config.rebalance_freq, bt_lib.algos.RunMonthly)

        algo_instance = algo_cls(**strategy_kwargs) if strategy_kwargs else algo_cls()

        strategy = bt_lib.Strategy(
            strategy_name,
            [
                rebal_cls(),
                bt_lib.algos.SelectAll(),
                algo_instance,
                bt_lib.algos.Rebalance(),
            ],
        )

        backtest = bt_lib.Backtest(strategy, data, initial_capital=self.config.initial_capital)

        logger.info(
            "Running portfolio backtest: strategy=%s, assets=%d, bars=%d",
            strategy_name, len(data.columns), len(data),
        )
        result = bt_lib.run(backtest)

        if _HAS_ENGINE:
            return self._convert_result(result, strategy_name)
        return result

    def _convert_result(self, bt_result, strategy_name: str) -> "BacktestResultV2":
        """Convert bt result to BacktestResultV2."""
        stats = bt_result.stats
        prices = bt_result.prices

        equity = prices[strategy_name] if strategy_name in prices.columns else prices.iloc[:, 0]
        equity_values = equity.values
        initial = self.config.initial_capital

        # bt returns rebased prices (100-based), not absolute equity.
        # Detect and convert back to absolute values.
        if len(equity_values) > 0 and abs(equity_values[0] - 100.0) < 1.0:
            equity_values = equity_values / 100.0 * initial

        total_return = (equity_values[-1] - initial) / initial
        n_days = max((equity.index[-1] - equity.index[0]).days, 1)
        n_years = n_days / 365.25
        final_ratio = max(0.0001, equity_values[-1] / initial)
        cagr = final_ratio ** (1 / n_years) - 1 if n_years > 0 else 0.0

        returns = equity.pct_change().dropna()
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        neg_ret = returns[returns < 0]
        if len(neg_ret) <= 1:
            downside = float(abs(neg_ret.mean())) if len(neg_ret) == 1 else 1e-10
        else:
            downside = float(neg_ret.std())
        if downside == 0:
            downside = 1e-10
        sortino = float(returns.mean() / downside * np.sqrt(252))

        peak = np.maximum.accumulate(equity_values)
        dd = (equity_values - peak) / peak
        max_dd = float(np.min(dd))
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Monthly returns
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        return BacktestResultV2(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            expectancy=0.0,
            avg_trade_duration=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            monthly_returns=monthly,
            alpha=0.0,
            beta=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
            equity_curve=equity,
            trade_log=[],
            trades=[],
            long_trades=0,
            short_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
        )
