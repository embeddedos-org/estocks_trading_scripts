"""
Factor Portfolio Strategy
===========================

Momentum long-short factor portfolio with monthly rebalancing.

Intermediate → Advanced example demonstrating:
- Cross-sectional momentum ranking
- Long top-N / short bottom-N portfolio construction
- Monthly rebalancing schedule
- Integration with FactorBacktester and BacktestEngineV2

Usage:
    from strategies.examples.factor_portfolio import FactorPortfolioStrategy
    strategy = FactorPortfolioStrategy()
    result = strategy.run_backtest(universe_prices)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.backtesting.backtest_engine_v2 import (
    BacktestContext,
    BacktestEngineV2,
    BacktestResultV2,
)
from strategies import register_strategy


@dataclass
class FactorPortfolioConfig:
    """Configuration for FactorPortfolioStrategy."""

    n_long: int = 5
    n_short: int = 5
    rebalance_freq: int = 21  # monthly
    momentum_lookback: int = 252
    momentum_skip: int = 21  # skip most recent month (12-1 month momentum)


@register_strategy("factor")
class FactorPortfolioStrategy:
    """Momentum long-short factor portfolio.

    Each month: rank universe by 12-1 month momentum.
    Long top N, short bottom N, equal weight within each leg.
    """

    def __init__(self, config: FactorPortfolioConfig | None = None) -> None:
        self.config = config or FactorPortfolioConfig()
        self._tickers: List[str] = []

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "FactorPortfolioStrategy":
        """Create strategy from a flat params dict (for optimizer)."""
        config = FactorPortfolioConfig(**{
            k: v for k, v in params.items() if hasattr(FactorPortfolioConfig, k)
        })
        return cls(config)

    def generate_signals(self, ctx: BacktestContext) -> Dict[str, int]:
        """Generate trading signals for the universe."""
        cfg = self.config
        signals: Dict[str, int] = {}

        if not self._tickers:
            self._tickers = list(ctx.bars.keys())

        # Only rebalance on schedule
        if ctx.bar_index % cfg.rebalance_freq != 0:
            # Maintain current positions
            for t in self._tickers:
                pos = ctx.positions.get(t, 0)
                signals[t] = 1 if pos > 0 else (-1 if pos < 0 else 0)
            return signals

        # Need enough history for momentum calculation
        min_bars = cfg.momentum_lookback + cfg.momentum_skip
        momentum_scores: Dict[str, float] = {}

        for t in self._tickers:
            if t not in ctx.bars or len(ctx.bars[t]) < min_bars:
                continue
            prices = ctx.bars[t]["close"]
            # 12-1 month momentum: return from (lookback+skip) ago to (skip) ago
            price_old = float(prices.iloc[-(cfg.momentum_lookback + cfg.momentum_skip)])
            price_recent = float(prices.iloc[-cfg.momentum_skip])
            if price_old > 0:
                momentum_scores[t] = (price_recent / price_old) - 1.0

        if len(momentum_scores) < cfg.n_long + cfg.n_short:
            return {t: 0 for t in self._tickers}

        # Rank and assign
        sorted_tickers = sorted(momentum_scores, key=momentum_scores.get, reverse=True)

        long_tickers = set(sorted_tickers[: cfg.n_long])
        short_tickers = set(sorted_tickers[-cfg.n_short:])

        for t in self._tickers:
            if t in long_tickers:
                signals[t] = 1
            elif t in short_tickers:
                signals[t] = -1
            else:
                signals[t] = 0

        return signals

    def run_backtest(
        self, universe_prices: pd.DataFrame, initial_capital: float = 100_000
    ) -> BacktestResultV2:
        """Run backtest on a universe of stocks.

        Args:
            universe_prices: DataFrame with columns as tickers, values as close prices.
            initial_capital: Starting capital.

        Returns:
            BacktestResultV2 with results.
        """
        tickers = list(universe_prices.columns)
        self._tickers = tickers

        data: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = pd.DataFrame({
                "close": universe_prices[t],
                "open": universe_prices[t],
                "high": universe_prices[t],
                "low": universe_prices[t],
                "volume": 1_000_000,
            })
            if "date" not in df.columns:
                df["date"] = universe_prices.index if isinstance(
                    universe_prices.index, pd.DatetimeIndex
                ) else pd.bdate_range("2020-01-01", periods=len(df))
            data[t] = df

        engine = BacktestEngineV2(initial_capital=initial_capital)
        engine.load_data(data)
        return engine.run(self.generate_signals)


def _generate_universe(
    n_stocks: int = 5, n_bars: int = 300, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic multi-stock universe."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_bars)
    tickers = [f"STOCK_{chr(65 + i)}" for i in range(n_stocks)]

    prices = {}
    for t in tickers:
        drift = rng.uniform(-0.0002, 0.001)
        vol = rng.uniform(0.01, 0.025)
        p = 50.0 + rng.uniform(0, 100)
        series = [p]
        for _ in range(n_bars - 1):
            p *= 1 + drift + rng.randn() * vol
            series.append(p)
        prices[t] = series

    return pd.DataFrame(prices, index=dates)


def run_example() -> BacktestResultV2:
    """Run factor portfolio strategy on synthetic data."""
    print("=" * 60)
    print("FACTOR PORTFOLIO STRATEGY EXAMPLE")
    print("=" * 60)

    universe = _generate_universe(n_stocks=5, n_bars=400)
    print(f"Universe: {list(universe.columns)}")
    print(f"Period: {universe.index[0].strftime('%Y-%m-%d')} to {universe.index[-1].strftime('%Y-%m-%d')}")
    print(f"Bars: {len(universe)}")

    strategy = FactorPortfolioStrategy(
        FactorPortfolioConfig(n_long=2, n_short=2)
    )
    result = strategy.run_backtest(universe)

    print(f"\nTotal Return:   {result.total_return:>10.2%}")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:>10.4f}")
    print(f"Max Drawdown:   {result.max_drawdown:>10.2%}")
    print(f"Total Trades:   {result.total_trades:>10d}")
    print(f"Long Trades:    {result.long_trades:>10d}")
    print(f"Short Trades:   {result.short_trades:>10d}")
    print(f"Win Rate:       {result.win_rate:>10.2%}")
    print(f"CAGR:           {result.cagr:>10.2%}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_example()
