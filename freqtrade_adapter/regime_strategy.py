"""
Regime-Aware Freqtrade Strategy
==================================

Concrete Freqtrade strategy that adapts to market regimes:
- TRENDING: EMA pullback entries with trailing stops
- RANGING: RSI + Bollinger Band mean reversion entries
- VOLATILE: No entries, tightened stops

Uses ML regime classifier when available, ADX-based fallback otherwise.
Ports logic from interactive_brokers/strategies/regime_trader.py.

Usage:
    freqtrade trade --config config_dry_run.json --strategy RegimeFreqtradeStrategy
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from freqtrade_adapter.strategy_adapter import StocksPluginStrategy


class RegimeFreqtradeStrategy(StocksPluginStrategy):
    """Regime-adaptive Freqtrade strategy.

    Classifies each candle's regime and applies the appropriate
    trading logic. Inherits all indicators from StocksPluginStrategy.

    Regime Thresholds (configurable via class attributes):
        adx_trend_threshold: ADX above this = TRENDING (default: 25)
        adx_range_threshold: ADX below this = RANGING (default: 20)
        atr_volatility_mult: ATR > avg * this = VOLATILE (default: 1.5)
    """

    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    atr_volatility_mult: float = 1.5
    atr_avg_window: int = 50

    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    ema_pullback_tolerance: float = 0.002

    minimal_roi = {
        "0": 0.08,
        "30": 0.04,
        "60": 0.02,
        "120": 0.01,
    }

    stoploss = -0.04

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """Compute indicators + regime classification.

        Adds regime column (0=TRENDING, 1=RANGING, 2=VOLATILE)
        and ML features if available.
        """
        dataframe = super().populate_indicators(dataframe, metadata)

        dataframe["atr_avg"] = dataframe["atr"].rolling(window=self.atr_avg_window).mean()

        dataframe["regime"] = 1  # default RANGING

        is_volatile = dataframe["atr"] > dataframe["atr_avg"] * self.atr_volatility_mult
        is_trending = dataframe["adx"] > self.adx_trend_threshold
        is_ranging = dataframe["adx"] < self.adx_range_threshold

        dataframe.loc[is_volatile, "regime"] = 2  # VOLATILE
        dataframe.loc[is_trending & ~is_volatile, "regime"] = 0  # TRENDING
        dataframe.loc[is_ranging & ~is_volatile, "regime"] = 1  # RANGING

        # ML regime classifier enrichment (optional)
        try:
            from shared.ml.regime_classifier import MLRegimeClassifier
            ml_features = MLRegimeClassifier.compute_features(dataframe)
            for col in ml_features.columns:
                if col not in dataframe.columns:
                    dataframe[f"ml_{col}"] = ml_features[col]
        except Exception:
            pass

        logger.debug(
            "Regime distribution for %s: TRENDING=%d, RANGING=%d, VOLATILE=%d",
            metadata.get("pair", "?"),
            (dataframe["regime"] == 0).sum(),
            (dataframe["regime"] == 1).sum(),
            (dataframe["regime"] == 2).sum(),
        )

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """Regime-aware entry signals.

        TRENDING: EMA pullback to fast EMA in direction of trend
        RANGING: RSI + BB extreme entries
        VOLATILE: No entries
        """
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0

        # ─── TRENDING REGIME: EMA Pullback Entries ───
        trend_uptrend = (
            (dataframe["regime"] == 0)
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["low"] <= dataframe["ema_9"] * (1 + self.ema_pullback_tolerance))
            & (dataframe["close"] > dataframe["ema_9"])
        )

        trend_downtrend = (
            (dataframe["regime"] == 0)
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["close"] < dataframe["ema_200"])
            & (dataframe["high"] >= dataframe["ema_9"] * (1 - self.ema_pullback_tolerance))
            & (dataframe["close"] < dataframe["ema_9"])
        )

        # ─── RANGING REGIME: Mean Reversion Entries ───
        range_long = (
            (dataframe["regime"] == 1)
            & (dataframe["rsi"] < self.rsi_oversold)
            & (dataframe["close"] <= dataframe["bb_lower"])
        )

        range_short = (
            (dataframe["regime"] == 1)
            & (dataframe["rsi"] > self.rsi_overbought)
            & (dataframe["close"] >= dataframe["bb_upper"])
        )

        # ─── VOLATILE: No entries (regime == 2 excluded) ───

        dataframe.loc[trend_uptrend | range_long, "enter_long"] = 1
        dataframe.loc[trend_downtrend | range_short, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """Regime-aware exit signals.

        TRENDING: Let trailing stop handle (Freqtrade config), RSI overbought exit
        RANGING: Exit at BB mid
        VOLATILE: Tighter exits — any adverse signal triggers exit
        """
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        # Trend exits: RSI extreme reversal
        trend_exit_long = (
            (dataframe["regime"] == 0)
            & (dataframe["rsi"] > 75)
            & (dataframe["ema_9"] < dataframe["ema_21"])
        )

        trend_exit_short = (
            (dataframe["regime"] == 0)
            & (dataframe["rsi"] < 25)
            & (dataframe["ema_9"] > dataframe["ema_21"])
        )

        # Ranging exits: price reverts to BB mid
        range_exit_long = (
            (dataframe["regime"] == 1)
            & (dataframe["close"] >= dataframe["bb_mid"])
        )

        range_exit_short = (
            (dataframe["regime"] == 1)
            & (dataframe["close"] <= dataframe["bb_mid"])
        )

        # Volatile exits: aggressive — exit on any adverse move
        volatile_exit_long = (
            (dataframe["regime"] == 2)
            & (dataframe["close"] < dataframe["close"].shift(1))
        )

        volatile_exit_short = (
            (dataframe["regime"] == 2)
            & (dataframe["close"] > dataframe["close"].shift(1))
        )

        dataframe.loc[
            trend_exit_long | range_exit_long | volatile_exit_long,
            "exit_long",
        ] = 1

        dataframe.loc[
            trend_exit_short | range_exit_short | volatile_exit_short,
            "exit_short",
        ] = 1

        return dataframe
