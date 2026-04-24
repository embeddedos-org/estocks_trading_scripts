"""
Trend Following Strategy
==========================

EMA crossover with ADX trend filter and ATR trailing stop.

Beginner → Intermediate example demonstrating:
- Dual EMA crossover for trend direction
- 200 EMA as regime filter (only long above, only short below)
- ADX filter to avoid choppy markets
- ATR-based position sizing and trailing stop

Usage:
    from strategies.examples.trend_following import TrendFollowingStrategy
    strategy = TrendFollowingStrategy()
    # Use with BacktestEngineV2
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

try:
    from shared.indicators.candlestick_patterns import CandlestickPatterns
    _HAS_CANDLE_PATTERNS = True
except ImportError:
    _HAS_CANDLE_PATTERNS = False
    CandlestickPatterns = None  # type: ignore[assignment,misc]

import logging

logger = logging.getLogger(__name__)


@dataclass
class TrendFollowingConfig:
    """Configuration for TrendFollowingStrategy."""

    fast_ma_length: int = 9
    slow_ma_length: int = 21
    trend_filter_length: int = 200
    use_adx_filter: bool = True
    adx_threshold: int = 25
    stop_loss_atr_mult: float = 2.0
    trailing_stop: bool = True
    use_volume_filter: bool = True
    volume_ma_length: int = 20
    use_mtf_filter: bool = True
    htf_period: str = "1D"
    use_candle_confirm: bool = False

    # Pyramiding (Livermore — adding to winners)
    enable_pyramiding: bool = False
    pyramid_threshold_pct: float = 2.0
    max_pyramid_adds: int = 3

    # Data enrichment (news, fundamentals, earnings, regime)
    use_enricher: bool = True


@register_strategy("trend_following")
class TrendFollowingStrategy:
    """EMA crossover with ADX filter and ATR trailing stop.

    Entry: fast EMA > slow EMA AND price > 200 EMA AND ADX > 25
    Exit: fast EMA < slow EMA OR trailing stop hit
    """

    def __init__(self, config: TrendFollowingConfig | None = None) -> None:
        self.config = config or TrendFollowingConfig()
        self._trailing_stops: Dict[str, float] = {}
        self._entry_prices: Dict[str, float] = {}
        self._pyramid_counts: Dict[str, int] = {}
        self._mtf = MultiTimeframeTrend(htf_period=self.config.htf_period)
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception:
                pass

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "TrendFollowingStrategy":
        """Create strategy from a flat params dict (for optimizer)."""
        config = TrendFollowingConfig(**{
            k: v for k, v in params.items() if hasattr(TrendFollowingConfig, k)
        })
        return cls(config)

    def generate_signals(self, ctx: BacktestContext) -> Dict[str, int]:
        """Generate trading signals for each symbol."""
        cfg = self.config
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < cfg.trend_filter_length:
                signals[sym] = 0
                continue

            close = df["close"]

            fast_ema = TI.ema(close, cfg.fast_ma_length)
            slow_ema = TI.ema(close, cfg.slow_ma_length)
            trend_ema = TI.ema(close, cfg.trend_filter_length)

            current_close = float(close.iloc[-1])
            fast_val = float(fast_ema.iloc[-1])
            slow_val = float(slow_ema.iloc[-1])
            trend_val = float(trend_ema.iloc[-1])

            adx_ok = True
            if cfg.use_adx_filter:
                adx_val, _, _ = TI.adx(df, 14)
                adx_ok = float(adx_val.iloc[-1]) > cfg.adx_threshold if not np.isnan(adx_val.iloc[-1]) else False

            atr = TI.atr(df, 14)
            current_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else current_close * 0.02

            # Volume filter: only generate new BUY/SELL signals when volume > SMA(volume, 20)
            volume_ok = True
            if cfg.use_volume_filter and "volume" in df.columns:
                vol = df["volume"]
                vol_sma = vol.rolling(cfg.volume_ma_length).mean()
                current_vol = float(vol.iloc[-1])
                vol_sma_val = float(vol_sma.iloc[-1]) if not np.isnan(vol_sma.iloc[-1]) else 0
                volume_ok = current_vol > vol_sma_val

            current_pos = ctx.positions.get(sym, 0)

            # Trailing stop logic
            if cfg.trailing_stop and current_pos != 0:
                if current_pos > 0:
                    new_stop = current_close - cfg.stop_loss_atr_mult * current_atr
                    self._trailing_stops[sym] = max(
                        self._trailing_stops.get(sym, 0), new_stop
                    )
                    if current_close < self._trailing_stops[sym]:
                        signals[sym] = 0
                        self._trailing_stops.pop(sym, None)
                        continue
                elif current_pos < 0:
                    new_stop = current_close + cfg.stop_loss_atr_mult * current_atr
                    self._trailing_stops[sym] = min(
                        self._trailing_stops.get(sym, float("inf")), new_stop
                    )
                    if current_close > self._trailing_stops[sym]:
                        signals[sym] = 0
                        self._trailing_stops.pop(sym, None)
                        continue

            # Candlestick confirmation helper
            candle_bullish = True
            candle_bearish = True
            if cfg.use_candle_confirm and _HAS_CANDLE_PATTERNS:
                recent = df.iloc[-3:]  # last 3 bars
                bullish_patterns = CandlestickPatterns.detect_bullish(recent)
                bearish_patterns = CandlestickPatterns.detect_bearish(recent)
                candle_bullish = len(bullish_patterns) > 0
                candle_bearish = len(bearish_patterns) > 0
                if not candle_bullish:
                    logger.debug("%s: no bullish candle confirmation in last 3 bars", sym)
                if not candle_bearish:
                    logger.debug("%s: no bearish candle confirmation in last 3 bars", sym)

            # Multi-timeframe trend filter
            mtf_buy_ok = True
            mtf_sell_ok = True
            if cfg.use_mtf_filter:
                mtf_buy_ok = self._mtf.is_aligned(df, "BUY")
                mtf_sell_ok = self._mtf.is_aligned(df, "SELL")

            # Entry / continuation logic
            # Enricher gate: check sentiment, fundamentals, earnings before new entry
            enricher_ok = True
            if getattr(self, "_enricher", None) and current_pos == 0:
                enriched = self._enricher.enrich(sym, df)
                blocked, reason = self._enricher.should_block_entry(enriched)
                if blocked:
                    enricher_ok = False
                    logger.debug("%s: entry blocked by enricher: %s", sym, reason)

            if fast_val > slow_val and current_close > trend_val and adx_ok and mtf_buy_ok and enricher_ok:
                if current_pos <= 0 and not volume_ok:
                    # New entry blocked by low volume
                    signals[sym] = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
                elif cfg.use_candle_confirm and not candle_bullish and current_pos <= 0:
                    signals[sym] = 0
                    logger.info("%s: BUY signal skipped — no bullish candle confirmation", sym)
                else:
                    signals[sym] = 1
                    if current_pos <= 0:
                        self._trailing_stops[sym] = current_close - cfg.stop_loss_atr_mult * current_atr
                        self._entry_prices[sym] = current_close
                        self._pyramid_counts[sym] = 0
            elif fast_val < slow_val and current_close < trend_val and adx_ok and mtf_sell_ok and enricher_ok:
                if current_pos >= 0 and not volume_ok:
                    # New entry blocked by low volume
                    signals[sym] = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
                elif cfg.use_candle_confirm and not candle_bearish and current_pos >= 0:
                    signals[sym] = 0
                    logger.info("%s: SELL signal skipped — no bearish candle confirmation", sym)
                else:
                    signals[sym] = -1
                    if current_pos >= 0:
                        self._trailing_stops[sym] = current_close + cfg.stop_loss_atr_mult * current_atr
                        self._entry_prices[sym] = current_close
                        self._pyramid_counts[sym] = 0
            elif fast_val < slow_val and current_pos > 0:
                signals[sym] = 0
                self._trailing_stops.pop(sym, None)
                self._entry_prices.pop(sym, None)
                self._pyramid_counts.pop(sym, None)
            elif fast_val > slow_val and current_pos < 0:
                signals[sym] = 0
                self._trailing_stops.pop(sym, None)
                self._entry_prices.pop(sym, None)
                self._pyramid_counts.pop(sym, None)
            else:
                # Pyramiding: add to winning positions
                if (
                    cfg.enable_pyramiding
                    and current_pos > 0
                    and sym in self._entry_prices
                ):
                    entry_price = self._entry_prices[sym]
                    pyramid_count = self._pyramid_counts.get(sym, 0)
                    unrealised_pct = (
                        (current_close - entry_price) / entry_price * 100
                        if entry_price > 0 else 0
                    )
                    threshold = cfg.pyramid_threshold_pct * (1 + pyramid_count)
                    if (
                        unrealised_pct >= threshold
                        and pyramid_count < cfg.max_pyramid_adds
                    ):
                        signals[sym] = 2  # signal to add to position
                        self._pyramid_counts[sym] = pyramid_count + 1
                        # Tighten trailing stop after each add
                        tighter_mult = cfg.stop_loss_atr_mult * (0.8 ** (pyramid_count + 1))
                        self._trailing_stops[sym] = current_close - tighter_mult * current_atr
                        logger.info(
                            "%s: PYRAMID ADD level %d (unrealised=%.1f%%, threshold=%.1f%%)",
                            sym, pyramid_count + 1, unrealised_pct, threshold,
                        )
                        continue
                signals[sym] = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

        return signals


def _generate_trending_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trending OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    price = 100.0
    prices = []
    for _ in range(n_bars):
        ret = 0.0005 + rng.randn() * 0.015  # slight upward drift
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.005)
        low = price * (1 - abs(rng.randn()) * 0.005)
        op = price * (1 + rng.randn() * 0.002)
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
    """Run trend following strategy on synthetic data."""
    print("=" * 60)
    print("TREND FOLLOWING STRATEGY EXAMPLE")
    print("=" * 60)

    df = _generate_trending_data()
    strategy = TrendFollowingStrategy()

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
    print(f"Calmar Ratio:   {result.calmar_ratio:>10.4f}")
    print(f"Expectancy:     ${result.expectancy:>9.2f}")
    print(f"Avg Duration:   {result.avg_trade_duration:>10.1f} bars")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_example()
