"""
Breakout Strategy
===================

Donchian channel breakout with volume confirmation and ATR trailing stop.

NEW — fills the gap: no Python breakout strategy existed before.
Translated from the TradingView momentum_breakout.pine concept.

Intermediate example demonstrating:
- Donchian channel (N-bar high/low) breakout detection
- Volume spike confirmation (volume > N × average)
- ATR-based trailing stop for exits
- Bar confirmation (close above channel for N bars)

Usage:
    from strategies.examples.breakout import BreakoutStrategy
    strategy = BreakoutStrategy()
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
from shared.indicators.multi_timeframe import MultiTimeframeTrend
from strategies import register_strategy


@dataclass
class BreakoutConfig:
    """Configuration for BreakoutStrategy."""

    channel_length: int = 20
    volume_mult: float = 1.5
    atr_stop_mult: float = 2.0
    confirm_bars: int = 1
    use_mtf_filter: bool = True
    htf_period: str = "1D"
    use_enricher: bool = True


@register_strategy("breakout")
class BreakoutStrategy:
    """Donchian channel breakout with volume confirmation.

    Long entry: close > 20-day high AND volume > 1.5× avg volume
    Short entry: close < 20-day low AND volume > 1.5× avg volume
    Exit: ATR trailing stop (entry ± 2×ATR)
    """

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        self.config = config or BreakoutConfig()
        self._trailing_stops: Dict[str, float] = {}
        self._breakout_bars: Dict[str, int] = {}
        self._mtf = MultiTimeframeTrend(htf_period=self.config.htf_period)
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception:
                pass

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "BreakoutStrategy":
        """Create strategy from a flat params dict (for optimizer)."""
        config = BreakoutConfig(**{
            k: v for k, v in params.items() if hasattr(BreakoutConfig, k)
        })
        return cls(config)

    def generate_signals(self, ctx: BacktestContext) -> Dict[str, int]:
        """Generate trading signals for each symbol."""
        cfg = self.config
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < cfg.channel_length + 5:
                signals[sym] = 0
                continue

            close = df["close"]
            volume = df["volume"]
            current_close = float(close.iloc[-1])
            current_volume = float(volume.iloc[-1])

            # Donchian channels (using lookback excluding current bar)
            dc = TI.donchian_channels(df, cfg.channel_length)
            dc_upper = float(dc["DCU"].iloc[-2]) if len(dc) > 1 else current_close
            dc_lower = float(dc["DCL"].iloc[-2]) if len(dc) > 1 else current_close

            # Volume filter
            vol_sma = TI.sma(volume.astype(float), 20)
            avg_vol = float(vol_sma.iloc[-1]) if not np.isnan(vol_sma.iloc[-1]) else current_volume
            volume_spike = current_volume > cfg.volume_mult * avg_vol

            # ATR for trailing stop
            atr = TI.atr(df, 14)
            current_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else current_close * 0.02

            current_pos = ctx.positions.get(sym, 0)

            # Trailing stop logic
            if current_pos > 0:
                new_stop = current_close - cfg.atr_stop_mult * current_atr
                self._trailing_stops[sym] = max(
                    self._trailing_stops.get(sym, 0), new_stop
                )
                if current_close < self._trailing_stops[sym]:
                    signals[sym] = 0
                    self._trailing_stops.pop(sym, None)
                    self._breakout_bars.pop(sym, None)
                    continue
            elif current_pos < 0:
                new_stop = current_close + cfg.atr_stop_mult * current_atr
                self._trailing_stops[sym] = min(
                    self._trailing_stops.get(sym, float("inf")), new_stop
                )
                if current_close > self._trailing_stops[sym]:
                    signals[sym] = 0
                    self._trailing_stops.pop(sym, None)
                    self._breakout_bars.pop(sym, None)
                    continue

            # Multi-timeframe trend filter: only take breakouts aligned with HTF
            mtf_buy_ok = True
            mtf_sell_ok = True
            if cfg.use_mtf_filter:
                mtf_buy_ok = self._mtf.is_aligned(df, "BUY")
                mtf_sell_ok = self._mtf.is_aligned(df, "SELL")

            # Entry logic with confirmation bars
            # Enricher gate: check sentiment, fundamentals, earnings before new entry
            enricher_ok = True
            if getattr(self, "_enricher", None) and current_pos == 0:
                enriched = self._enricher.enrich(sym, df)
                blocked, reason = self._enricher.should_block_entry(enriched)
                if blocked:
                    enricher_ok = False

            if current_pos == 0:
                # Long breakout
                if current_close > dc_upper and volume_spike and mtf_buy_ok and enricher_ok:
                    bars_above = self._breakout_bars.get(sym, 0) + 1
                    self._breakout_bars[sym] = bars_above
                    if bars_above >= cfg.confirm_bars:
                        signals[sym] = 1
                        self._trailing_stops[sym] = current_close - cfg.atr_stop_mult * current_atr
                        self._breakout_bars.pop(sym, None)
                    else:
                        signals[sym] = 0
                # Short breakout
                elif current_close < dc_lower and volume_spike and mtf_sell_ok:
                    bars_below = self._breakout_bars.get(sym, 0) + 1
                    self._breakout_bars[sym] = bars_below
                    if bars_below >= cfg.confirm_bars:
                        signals[sym] = -1
                        self._trailing_stops[sym] = current_close + cfg.atr_stop_mult * current_atr
                        self._breakout_bars.pop(sym, None)
                    else:
                        signals[sym] = 0
                else:
                    self._breakout_bars.pop(sym, None)
                    signals[sym] = 0
            else:
                # Hold position (trailing stop handles exit)
                signals[sym] = 1 if current_pos > 0 else -1

        return signals


def _generate_breakout_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with consolidation and breakout patterns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    price = 100.0
    prices = []
    regime = "range"  # "range" or "trend"
    regime_bars = 0

    for i in range(n_bars):
        regime_bars += 1
        if regime == "range" and regime_bars > 40 and rng.random() < 0.08:
            regime = "trend"
            regime_bars = 0
        elif regime == "trend" and regime_bars > 20 and rng.random() < 0.15:
            regime = "range"
            regime_bars = 0

        if regime == "range":
            ret = rng.randn() * 0.008  # low volatility consolidation
        else:
            ret = 0.003 + rng.randn() * 0.018  # trending with higher vol

        price *= 1 + ret
        spread = abs(rng.randn()) * 0.006 + 0.002
        high = price * (1 + spread)
        low = price * (1 - spread)
        op = price * (1 + rng.randn() * 0.003)

        # Volume spikes on breakouts
        base_vol = 1_000_000
        vol_mult = 2.5 if regime == "trend" and regime_bars < 3 else 1.0
        vol = int(base_vol * vol_mult * rng.uniform(0.7, 1.3))

        prices.append({
            "date": dates[i],
            "open": op,
            "high": high,
            "low": low,
            "close": price,
            "volume": vol,
        })
    return pd.DataFrame(prices)


def run_example() -> BacktestResultV2:
    """Run breakout strategy on synthetic data."""
    print("=" * 60)
    print("BREAKOUT STRATEGY EXAMPLE")
    print("=" * 60)

    df = _generate_breakout_data()
    strategy = BreakoutStrategy()

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
