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
    stop_loss_pct: float = 0.05  # 5% stop loss
    atr_stop_multiplier: float = 2.0  # ATR-based stop price multiplier
    use_enricher: bool = True


@register_strategy("factor")
class FactorPortfolioStrategy:
    """Momentum long-short factor portfolio.

    Each month: rank universe by 12-1 month momentum.
    Long top N, short bottom N, equal weight within each leg.
    """

    def __init__(self, config: FactorPortfolioConfig | None = None) -> None:
        self.config = config or FactorPortfolioConfig()
        self._tickers: List[str] = []
        self._entry_prices: Dict[str, float] = {}
        self._stop_prices: Dict[str, float] = {}
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception:
                pass

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

        # Check stop losses on all held positions first
        for t in self._tickers:
            pos = ctx.positions.get(t, 0)
            if pos != 0 and t in ctx.bars and len(ctx.bars[t]) > 0:
                current_price = float(ctx.bars[t]["close"].iloc[-1])

                # ATR-based stop price check
                if t in self._stop_prices:
                    if pos > 0 and current_price < self._stop_prices[t]:
                        signals[t] = 0
                        self._entry_prices.pop(t, None)
                        self._stop_prices.pop(t, None)
                        continue
                    elif pos < 0 and current_price > self._stop_prices[t]:
                        signals[t] = 0
                        self._entry_prices.pop(t, None)
                        self._stop_prices.pop(t, None)
                        continue

                # Percentage-based stop loss check
                if t in self._entry_prices:
                    entry = self._entry_prices[t]
                    if pos > 0 and current_price < entry * (1 - cfg.stop_loss_pct):
                        signals[t] = 0
                        self._entry_prices.pop(t, None)
                        self._stop_prices.pop(t, None)
                        continue
                    elif pos < 0 and current_price > entry * (1 + cfg.stop_loss_pct):
                        signals[t] = 0
                        self._entry_prices.pop(t, None)
                        self._stop_prices.pop(t, None)
                        continue

        # Only rebalance on schedule
        if ctx.bar_index % cfg.rebalance_freq != 0:
            # Maintain current positions (unless stopped out above)
            for t in self._tickers:
                if t not in signals:
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
            return {t: signals.get(t, 0) for t in self._tickers}

        # Rank and assign
        sorted_tickers = sorted(momentum_scores, key=lambda k: momentum_scores.get(k, 0.0), reverse=True)

        long_tickers = set(sorted_tickers[: cfg.n_long])
        short_tickers = set(sorted_tickers[-cfg.n_short:])

        for t in self._tickers:
            if t in signals:
                continue  # already stopped out

            current_price = float(ctx.bars[t]["close"].iloc[-1]) if t in ctx.bars and len(ctx.bars[t]) > 0 else 0
            prev_pos = ctx.positions.get(t, 0)

            if t in long_tickers:
                # Enricher gate
                enricher_ok = True
                if getattr(self, "_enricher", None) and prev_pos <= 0 and t in ctx.bars:
                    enriched = self._enricher.enrich(t, ctx.bars[t])
                    blocked, _ = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False
                if enricher_ok:
                    signals[t] = 1
                    if prev_pos <= 0:
                        self._entry_prices[t] = current_price
                        atr_value = self._compute_atr(ctx.bars[t])
                        self._stop_prices[t] = current_price - cfg.atr_stop_multiplier * atr_value
                else:
                    signals[t] = 0
            elif t in short_tickers:
                enricher_ok = True
                if getattr(self, "_enricher", None) and prev_pos >= 0 and t in ctx.bars:
                    enriched = self._enricher.enrich(t, ctx.bars[t])
                    blocked, _ = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False
                    # Also block shorts when sentiment is strongly bullish
                    if enriched.sentiment_available and enriched.sentiment_score > 0.4:
                        enricher_ok = False
                if enricher_ok:
                    signals[t] = -1
                    if prev_pos >= 0:
                        self._entry_prices[t] = current_price
                        atr_value = self._compute_atr(ctx.bars[t])
                        self._stop_prices[t] = current_price + cfg.atr_stop_multiplier * atr_value
                else:
                    signals[t] = 0
            else:
                signals[t] = 0
                self._entry_prices.pop(t, None)
                self._stop_prices.pop(t, None)

        return signals

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR for stop loss calculation."""
        if len(df) < period + 1:
            return float(df["close"].iloc[-1]) * 0.02  # fallback 2%
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        val = float(atr.iloc[-1])
        return val if not np.isnan(val) else float(close.iloc[-1]) * 0.02

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
