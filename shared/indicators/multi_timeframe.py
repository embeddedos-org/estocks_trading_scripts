"""Multi-timeframe trend confirmation module."""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class MultiTimeframeTrend:
    """Provides higher-timeframe trend context for intraday strategies."""

    def __init__(self, htf_period: str = "1D", sma_fast: int = 20, sma_slow: int = 50):
        self.htf_period = htf_period
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow

    def resample_to_htf(self, df: pd.DataFrame, period: str = None) -> pd.DataFrame:
        """Resample intraday OHLCV data to higher timeframe."""
        period = period or self.htf_period
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        htf = df.resample(period).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()
        return htf

    def get_htf_trend(self, df: pd.DataFrame) -> str:
        """Returns 'BULLISH', 'BEARISH', or 'NEUTRAL' based on higher timeframe."""
        htf = self.resample_to_htf(df)
        if len(htf) < self.sma_slow:
            return "NEUTRAL"
        fast_sma = htf["close"].rolling(self.sma_fast).mean().iloc[-1]
        slow_sma = htf["close"].rolling(self.sma_slow).mean().iloc[-1]
        price = htf["close"].iloc[-1]
        if price > fast_sma > slow_sma:
            return "BULLISH"
        elif price < fast_sma < slow_sma:
            return "BEARISH"
        return "NEUTRAL"

    def is_aligned(self, df: pd.DataFrame, direction: str) -> bool:
        """Check if the higher-timeframe trend aligns with the desired trade direction."""
        trend = self.get_htf_trend(df)
        if direction.upper() == "BUY":
            return trend in ("BULLISH", "NEUTRAL")
        elif direction.upper() == "SELL":
            return trend in ("BEARISH", "NEUTRAL")
        return True

    def get_htf_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> dict:
        """Calculate higher-timeframe support/resistance levels."""
        htf = self.resample_to_htf(df)
        if len(htf) < lookback:
            return {"support": 0, "resistance": float("inf")}
        recent = htf.tail(lookback)
        return {
            "support": float(recent["low"].min()),
            "resistance": float(recent["high"].max()),
            "pivot": float((recent["high"].iloc[-1] + recent["low"].iloc[-1] + recent["close"].iloc[-1]) / 3)
        }
