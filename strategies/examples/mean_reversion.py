"""
Mean Reversion Strategy
=========================

RSI + Bollinger Band mean reversion with confirmation.

Beginner → Intermediate example demonstrating:
- RSI oversold/overbought detection
- Bollinger Band price extreme confirmation
- Hard stop loss for risk management
- Mean (BB midline) reversion targets

Usage:
    from strategies.examples.mean_reversion import MeanReversionStrategy
    strategy = MeanReversionStrategy()
    engine = BacktestEngineV2()
    engine.load_data(df)
    result = engine.run(strategy.generate_signals)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.backtesting.backtest_engine_v2 import (
    BacktestContext,
    BacktestEngineV2,
    BacktestResultV2,
)
from shared.indicators.technical_indicators import TechnicalIndicators as TI
from strategies import register_strategy


@dataclass
class MeanReversionConfig:
    """Configuration for MeanReversionStrategy."""

    rsi_length: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bb_length: int = 20
    bb_std: float = 2.0
    use_bb_confirm: bool = True
    stop_loss_pct: float = 0.03


@register_strategy("mean_reversion")
class MeanReversionStrategy:
    """RSI + Bollinger Band mean reversion strategy.

    Long entry: RSI < 30 AND price <= BB lower band
    Short entry: RSI > 70 AND price >= BB upper band
    Exit: price crosses BB midline OR stop loss
    """

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()
        self._entry_prices: Dict[str, float] = {}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MeanReversionStrategy":
        """Create strategy from a flat params dict (for optimizer)."""
        config = MeanReversionConfig(**{
            k: v for k, v in params.items() if hasattr(MeanReversionConfig, k)
        })
        return cls(config)

    def generate_signals(self, ctx: BacktestContext) -> Dict[str, int]:
        """Generate trading signals for each symbol."""
        cfg = self.config
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < max(cfg.rsi_length, cfg.bb_length) + 5:
                signals[sym] = 0
                continue

            close = df["close"]
            current_close = float(close.iloc[-1])

            rsi = TI.rsi(close, cfg.rsi_length)
            current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

            bb = TI.bbands(close, cfg.bb_length, cfg.bb_std)
            bb_lower = float(bb["BBL"].iloc[-1])
            bb_mid = float(bb["BBM"].iloc[-1])
            bb_upper = float(bb["BBU"].iloc[-1])

            current_pos = ctx.positions.get(sym, 0)

            # Stop loss check
            if current_pos != 0 and sym in self._entry_prices:
                entry = self._entry_prices[sym]
                if current_pos > 0 and current_close < entry * (1 - cfg.stop_loss_pct):
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    continue
                if current_pos < 0 and current_close > entry * (1 + cfg.stop_loss_pct):
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    continue

            # Exit at BB midline (mean reversion target)
            if current_pos > 0 and current_close >= bb_mid:
                signals[sym] = 0
                self._entry_prices.pop(sym, None)
                continue
            if current_pos < 0 and current_close <= bb_mid:
                signals[sym] = 0
                self._entry_prices.pop(sym, None)
                continue

            # Entry logic
            if current_pos == 0:
                bb_long_ok = current_close <= bb_lower if cfg.use_bb_confirm else True
                bb_short_ok = current_close >= bb_upper if cfg.use_bb_confirm else True

                if current_rsi < cfg.rsi_oversold and bb_long_ok:
                    signals[sym] = 1
                    self._entry_prices[sym] = current_close
                elif current_rsi > cfg.rsi_overbought and bb_short_ok:
                    signals[sym] = -1
                    self._entry_prices[sym] = current_close
                else:
                    signals[sym] = 0
            else:
                # Hold existing position
                signals[sym] = 1 if current_pos > 0 else -1

        return signals


def _generate_ranging_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic mean-reverting (ranging) OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    mean_price = 100.0
    price = mean_price
    prices = []
    for _ in range(n_bars):
        # Mean-reverting process (Ornstein-Uhlenbeck)
        ret = 0.05 * (mean_price - price) / mean_price + rng.randn() * 0.015
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.008)
        low = price * (1 - abs(rng.randn()) * 0.008)
        op = price * (1 + rng.randn() * 0.003)
        prices.append({
            "date": dates[len(prices)],
            "open": op,
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(prices)


def run_example() -> BacktestResultV2:
    """Run mean reversion strategy on synthetic data."""
    print("=" * 60)
    print("MEAN REVERSION STRATEGY EXAMPLE")
    print("=" * 60)

    df = _generate_ranging_data()
    strategy = MeanReversionStrategy()

    engine = BacktestEngineV2(initial_capital=100_000)
    engine.load_data(df)
    result = engine.run(strategy.generate_signals)

    print(f"\nTotal Return:   {result.total_return:>10.2%}")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:>10.4f}")
    print(f"Sortino Ratio:  {result.sortino_ratio:>10.4f}")
    print(f"Max Drawdown:   {result.max_drawdown:>10.2%}")
    print(f"Win Rate:       {result.win_rate:>10.2%}")
    print(f"Profit Factor:  {result.profit_factor:>10.4f}")
    print(f"Total Trades:   {result.total_trades:>10d}")
    print(f"CAGR:           {result.cagr:>10.2%}")
    print(f"Expectancy:     ${result.expectancy:>9.2f}")
    print(f"Avg Duration:   {result.avg_trade_duration:>10.1f} bars")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_example()
