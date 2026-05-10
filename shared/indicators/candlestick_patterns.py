"""
Candlestick Pattern Recognition
=================================

TA-Lib C-accelerated candlestick patterns with manual fallbacks
for basic patterns (doji, hammer, engulfing).

Usage:
    from shared.indicators.candlestick_patterns import CandlestickPatterns as CP
    signals = CP.scan_all(df)
    doji = CP.doji(df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib  # type: ignore[import-untyped]
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False


def _require_talib(pattern_name: str) -> None:
    """Raise ImportError with install instructions for complex patterns."""
    if not _HAS_TALIB:
        raise ImportError(
            f"CandlestickPatterns.{pattern_name}() requires TA-Lib. "
            "Install: conda install -c conda-forge ta-lib"
        )


class CandlestickPatterns:
    """Static-method library for candlestick pattern detection.

    Simple patterns (doji, hammer, engulfing) have manual fallbacks.
    Complex patterns require TA-Lib.
    """

    @staticmethod
    def doji(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
        """Detect Doji candles (open ~= close).

        Args:
            df: OHLCV DataFrame
            threshold: Max body/range ratio to qualify as doji (default 5%)

        Returns:
            Series of 0/100 signals (100 = doji detected)
        """
        if _HAS_TALIB:
            return pd.Series(
                talib.CDLDOJI(
                    df["open"].values.astype(float),
                    df["high"].values.astype(float),
                    df["low"].values.astype(float),
                    df["close"].values.astype(float),
                ),
                index=df.index, name="DOJI",
            )
        body = (df["close"] - df["open"]).abs()
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        is_doji = (body / hl_range) < threshold
        return (is_doji.astype(int) * 100).rename("DOJI")

    @staticmethod
    def hammer(df: pd.DataFrame, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
        """Detect Hammer candles (small body at top, long lower shadow).

        Returns:
            Series of 0/100 signals (100 = hammer detected)
        """
        if _HAS_TALIB:
            return pd.Series(
                talib.CDLHAMMER(
                    df["open"].values.astype(float),
                    df["high"].values.astype(float),
                    df["low"].values.astype(float),
                    df["close"].values.astype(float),
                ),
                index=df.index, name="HAMMER",
            )
        body = (df["close"] - df["open"]).abs()
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        lower_shadow = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
        upper_shadow = df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
        is_hammer = (
            (body / hl_range < body_ratio)
            & (lower_shadow > shadow_ratio * body)
            & (upper_shadow < body)
        )
        return (is_hammer.astype(int) * 100).rename("HAMMER")

    @staticmethod
    def engulfing(df: pd.DataFrame) -> pd.Series:
        """Detect Bullish/Bearish Engulfing patterns.

        Returns:
            Series: +100 = bullish engulfing, -100 = bearish engulfing, 0 = none
        """
        if _HAS_TALIB:
            return pd.Series(
                talib.CDLENGULFING(
                    df["open"].values.astype(float),
                    df["high"].values.astype(float),
                    df["low"].values.astype(float),
                    df["close"].values.astype(float),
                ),
                index=df.index, name="ENGULFING",
            )
        prev_open = df["open"].shift(1)
        prev_close = df["close"].shift(1)
        curr_open = df["open"]
        curr_close = df["close"]

        bullish = (
            (prev_close < prev_open)  # prev bearish
            & (curr_close > curr_open)  # curr bullish
            & (curr_open <= prev_close)
            & (curr_close >= prev_open)
        )
        bearish = (
            (prev_close > prev_open)  # prev bullish
            & (curr_close < curr_open)  # curr bearish
            & (curr_open >= prev_close)
            & (curr_close <= prev_open)
        )
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[bullish] = 100
        signal[bearish] = -100
        return signal.rename("ENGULFING")

    @staticmethod
    def morning_star(df: pd.DataFrame) -> pd.Series:
        """Detect Morning Star pattern (bullish reversal). Requires TA-Lib."""
        _require_talib("morning_star")
        return pd.Series(
            talib.CDLMORNINGSTAR(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="MORNING_STAR",
        )

    @staticmethod
    def evening_star(df: pd.DataFrame) -> pd.Series:
        """Detect Evening Star pattern (bearish reversal). Requires TA-Lib."""
        _require_talib("evening_star")
        return pd.Series(
            talib.CDLEVENINGSTAR(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="EVENING_STAR",
        )

    @staticmethod
    def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
        """Detect Three White Soldiers (bullish). Requires TA-Lib."""
        _require_talib("three_white_soldiers")
        return pd.Series(
            talib.CDL3WHITESOLDIERS(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="THREE_WHITE_SOLDIERS",
        )

    @staticmethod
    def three_black_crows(df: pd.DataFrame) -> pd.Series:
        """Detect Three Black Crows (bearish). Requires TA-Lib."""
        _require_talib("three_black_crows")
        return pd.Series(
            talib.CDL3BLACKCROWS(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="THREE_BLACK_CROWS",
        )

    @staticmethod
    def harami(df: pd.DataFrame) -> pd.Series:
        """Detect Harami pattern. Requires TA-Lib."""
        _require_talib("harami")
        return pd.Series(
            talib.CDLHARAMI(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="HARAMI",
        )

    @staticmethod
    def shooting_star(df: pd.DataFrame) -> pd.Series:
        """Detect Shooting Star (bearish reversal). Requires TA-Lib."""
        _require_talib("shooting_star")
        return pd.Series(
            talib.CDLSHOOTINGSTAR(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="SHOOTING_STAR",
        )

    @staticmethod
    def hanging_man(df: pd.DataFrame) -> pd.Series:
        """Detect Hanging Man (bearish reversal). Requires TA-Lib."""
        _require_talib("hanging_man")
        return pd.Series(
            talib.CDLHANGINGMAN(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="HANGING_MAN",
        )

    @staticmethod
    def spinning_top(df: pd.DataFrame) -> pd.Series:
        """Detect Spinning Top. Requires TA-Lib."""
        _require_talib("spinning_top")
        return pd.Series(
            talib.CDLSPINNINGTOP(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="SPINNING_TOP",
        )

    @staticmethod
    def marubozu(df: pd.DataFrame) -> pd.Series:
        """Detect Marubozu (strong body, no shadows). Requires TA-Lib."""
        _require_talib("marubozu")
        return pd.Series(
            talib.CDLMARUBOZU(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            ),
            index=df.index, name="MARUBOZU",
        )

    @staticmethod
    def cup_and_handle(
        df: pd.DataFrame, cup_len: int = 30, handle_len: int = 10,
        max_scan_bars: int = 500,
    ) -> pd.Series:
        """Detect Cup and Handle pattern (bullish continuation).

        Detection logic:
        1. Find swing high (left lip), followed by a decline of 15-35%
        2. Base forms a rounded bottom over ``cup_len`` bars
        3. Price recovers to within 5% of the left lip (right lip)
        4. Handle: small pullback (< 12% from right lip) over ``handle_len`` bars
        5. Breakout: price closes above right lip on above-average volume

        Args:
            df: OHLCV DataFrame with 'high', 'low', 'close', 'volume' columns.
            cup_len: Minimum bars for cup formation (default 30).
            handle_len: Maximum bars for handle formation (default 10).
            max_scan_bars: Maximum bars to scan from the end (default 500).
                Prevents O(n²) performance on large intraday datasets.

        Returns:
            Series of 0/100 signals (100 = cup-and-handle breakout detected).
        """
        n = len(df)
        signal = pd.Series(0, index=df.index, dtype=int, name="CUP_AND_HANDLE")
        if n < cup_len + handle_len + 5:
            return signal

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(n)

        avg_volume = pd.Series(volume).rolling(window=20, min_periods=1).mean().values

        lookback = cup_len + handle_len + 5
        scan_start = max(lookback, n - max_scan_bars)
        for i in range(scan_start, n):
            window_start = i - lookback
            window_close = close[window_start:i + 1]
            window_high = high[window_start:i + 1]

            left_lip_idx = int(np.argmax(window_high[:cup_len // 2]))
            left_lip_price = window_high[left_lip_idx]

            if left_lip_price <= 0:
                continue

            cup_region = window_close[left_lip_idx:left_lip_idx + cup_len]
            if len(cup_region) < cup_len // 2:
                continue

            cup_low = float(np.min(cup_region))
            decline_pct = (left_lip_price - cup_low) / left_lip_price

            if decline_pct < 0.12 or decline_pct > 0.40:
                continue

            right_lip_region = window_close[-handle_len - 5:-handle_len] if handle_len < len(window_close) - 5 else window_close[-5:]
            if len(right_lip_region) == 0:
                continue
            right_lip_price = float(np.max(right_lip_region))

            recovery_pct = abs(right_lip_price - left_lip_price) / left_lip_price
            if recovery_pct > 0.08:
                continue

            handle_region = window_close[-handle_len:]
            handle_low = float(np.min(handle_region))
            handle_pullback = (right_lip_price - handle_low) / right_lip_price if right_lip_price > 0 else 1.0
            if handle_pullback > 0.12:
                continue

            current_close = close[i]
            if current_close > right_lip_price and volume[i] > avg_volume[i]:
                signal.iloc[i] = 100

        return signal

    @staticmethod
    def scan_all(df: pd.DataFrame) -> pd.DataFrame:
        """Run all available candlestick patterns and return detected signals.

        Returns:
            DataFrame with one column per pattern, values are signal strengths.
            Only patterns whose dependencies are met are included.
        """
        results = {}

        # Always available (have manual fallbacks)
        results["DOJI"] = CandlestickPatterns.doji(df)
        results["HAMMER"] = CandlestickPatterns.hammer(df)
        results["ENGULFING"] = CandlestickPatterns.engulfing(df)
        results["CUP_AND_HANDLE"] = CandlestickPatterns.cup_and_handle(df)

        # TA-Lib only patterns
        if _HAS_TALIB:
            talib_patterns = [
                ("MORNING_STAR", CandlestickPatterns.morning_star),
                ("EVENING_STAR", CandlestickPatterns.evening_star),
                ("THREE_WHITE_SOLDIERS", CandlestickPatterns.three_white_soldiers),
                ("THREE_BLACK_CROWS", CandlestickPatterns.three_black_crows),
                ("HARAMI", CandlestickPatterns.harami),
                ("SHOOTING_STAR", CandlestickPatterns.shooting_star),
                ("HANGING_MAN", CandlestickPatterns.hanging_man),
                ("SPINNING_TOP", CandlestickPatterns.spinning_top),
                ("MARUBOZU", CandlestickPatterns.marubozu),
            ]
            for name, func in talib_patterns:
                try:
                    results[name] = func(df)
                except Exception as e:
                    logger.warning("Pattern %s failed: %s", name, e)

        return pd.DataFrame(results)
