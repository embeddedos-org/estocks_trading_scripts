"""
Freqtrade Strategy Adapter
============================

Base adapter class that bridges stocks_plugin's indicator library
and strategies into Freqtrade's IStrategy interface. Subclass this
to create concrete Freqtrade strategies that reuse all 30+ indicators.

Requires: pip install freqtrade

Usage:
    # In freqtrade config, set:
    #   "strategy": "StocksPluginStrategy"
    # Or subclass for custom logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from freqtrade.strategy import IStrategy, merge_informative_pair  # type: ignore[import-untyped]
    from freqtrade.strategy import (  # type: ignore[import-untyped]
        BooleanParameter,
        DecimalParameter,
        IntParameter,
    )
    _HAS_FREQTRADE = True
except ImportError:
    _HAS_FREQTRADE = False
    logger.debug("freqtrade not installed — adapter unavailable")

    class IStrategy:  # type: ignore[no-redef]
        """Stub for when freqtrade is not installed."""
        timeframe = "5m"
        def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            return dataframe
        def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            return dataframe
        def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            return dataframe

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.indicators.technical_indicators import TechnicalIndicators as TI


class StocksPluginStrategy(IStrategy):
    """Base Freqtrade strategy adapter using stocks_plugin indicators.

    Computes all indicators via TechnicalIndicators and provides
    template entry/exit methods. Subclass to implement specific logic.

    Configuration:
        timeframe: Trading timeframe (default: "5m").
        minimal_roi: ROI table for auto-closing positions.
        stoploss: Global stoploss as negative decimal.
        trailing_stop: Enable trailing stop.
    """

    timeframe = "5m"

    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.025,
        "120": 0.01,
    }

    stoploss = -0.05

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    startup_candle_count: int = 200

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """Compute all technical indicators on the dataframe.

        Adds RSI, MACD, Bollinger Bands, ADX, Stochastic, ATR,
        EMAs (9/21/200), and more using shared TechnicalIndicators.

        Args:
            dataframe: OHLCV DataFrame from Freqtrade.
            metadata: Pair metadata dict.

        Returns:
            DataFrame enriched with indicator columns.
        """
        close = dataframe["close"]
        df_ohlcv = dataframe[["open", "high", "low", "close", "volume"]].copy()

        # Trend indicators
        dataframe["ema_9"] = TI.ema(close, 9)
        dataframe["ema_21"] = TI.ema(close, 21)
        dataframe["ema_50"] = TI.ema(close, 50)
        dataframe["ema_200"] = TI.ema(close, 200)
        dataframe["sma_20"] = TI.sma(close, 20)

        # Momentum
        dataframe["rsi"] = TI.rsi(close, 14)
        macd_line, signal_line, histogram = TI.macd(close)
        dataframe["macd"] = macd_line
        dataframe["macd_signal"] = signal_line
        dataframe["macd_hist"] = histogram

        stoch_k, stoch_d = TI.stochastic(df_ohlcv)
        dataframe["stoch_k"] = stoch_k
        dataframe["stoch_d"] = stoch_d

        # Volatility
        bb = TI.bbands(close)
        dataframe["bb_upper"] = bb["BBU"]
        dataframe["bb_mid"] = bb["BBM"]
        dataframe["bb_lower"] = bb["BBL"]
        dataframe["bb_pct_b"] = bb["BBP"]
        dataframe["bb_width"] = bb["BBB"]

        dataframe["atr"] = TI.atr(df_ohlcv, 14)

        # Directional
        adx_val, plus_di, minus_di = TI.adx(df_ohlcv)
        dataframe["adx"] = adx_val
        dataframe["plus_di"] = plus_di
        dataframe["minus_di"] = minus_di

        # Volume
        dataframe["obv"] = TI.obv(df_ohlcv)

        logger.debug(
            "Indicators computed for %s: %d columns",
            metadata.get("pair", "?"),
            len(dataframe.columns),
        )

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """Default entry logic: regime-based entries.

        - Trending regime (ADX > 25): EMA crossover entries
        - Ranging regime (ADX < 20): RSI extreme entries

        Subclasses should override this for custom logic.

        Args:
            dataframe: DataFrame with indicators.
            metadata: Pair metadata dict.

        Returns:
            DataFrame with 'enter_long' and 'enter_short' columns.
        """
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0

        # Trending long: EMA9 crosses above EMA21, above EMA200, ADX > 25
        trending_long = (
            (dataframe["adx"] > 25)
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["ema_9"].shift(1) <= dataframe["ema_21"].shift(1))
            & (dataframe["close"] > dataframe["ema_200"])
        )

        # Ranging long: RSI oversold + at lower BB
        ranging_long = (
            (dataframe["adx"] < 20)
            & (dataframe["rsi"] < 30)
            & (dataframe["close"] <= dataframe["bb_lower"])
        )

        dataframe.loc[trending_long | ranging_long, "enter_long"] = 1

        # Trending short: EMA9 crosses below EMA21, below EMA200
        trending_short = (
            (dataframe["adx"] > 25)
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["ema_9"].shift(1) >= dataframe["ema_21"].shift(1))
            & (dataframe["close"] < dataframe["ema_200"])
        )

        # Ranging short: RSI overbought + at upper BB
        ranging_short = (
            (dataframe["adx"] < 20)
            & (dataframe["rsi"] > 70)
            & (dataframe["close"] >= dataframe["bb_upper"])
        )

        dataframe.loc[trending_short | ranging_short, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """Default exit logic.

        - Trend exits: ATR trailing stop (via Freqtrade trailing_stop config)
        - Mean reversion exits: price returns to BB mid

        Args:
            dataframe: DataFrame with indicators.
            metadata: Pair metadata dict.

        Returns:
            DataFrame with 'exit_long' and 'exit_short' columns.
        """
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        # Exit long when price crosses above BB mid in ranging regime
        exit_long_mr = (
            (dataframe["adx"] < 20)
            & (dataframe["close"] >= dataframe["bb_mid"])
            & (dataframe["close"].shift(1) < dataframe["bb_mid"].shift(1))
        )

        # Exit long when RSI overbought in trending regime
        exit_long_trend = (
            (dataframe["adx"] > 25)
            & (dataframe["rsi"] > 70)
        )

        dataframe.loc[exit_long_mr | exit_long_trend, "exit_long"] = 1

        # Exit short when price crosses below BB mid
        exit_short_mr = (
            (dataframe["adx"] < 20)
            & (dataframe["close"] <= dataframe["bb_mid"])
            & (dataframe["close"].shift(1) > dataframe["bb_mid"].shift(1))
        )

        exit_short_trend = (
            (dataframe["adx"] > 25)
            & (dataframe["rsi"] < 30)
        )

        dataframe.loc[exit_short_mr | exit_short_trend, "exit_short"] = 1

        return dataframe
