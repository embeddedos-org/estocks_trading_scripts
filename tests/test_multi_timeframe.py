"""Tests for the MultiTimeframeTrend module.

Covers resampling intraday→daily, trend direction detection (BULLISH,
BEARISH, NEUTRAL), trend alignment checks, support/resistance
calculation, edge cases (insufficient data, non-datetime index).

15+ tests total.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.indicators.multi_timeframe import MultiTimeframeTrend


@pytest.fixture
def mtf():
    """Default MultiTimeframeTrend with fast=20, slow=50."""
    return MultiTimeframeTrend(htf_period="1D", sma_fast=20, sma_slow=50)


def _make_intraday_df(n_days=80, bars_per_day=78, trend="up"):
    """Create intraday OHLCV with a clear directional trend.

    Args:
        n_days: Number of trading days.
        bars_per_day: 5-min bars per day (6.5 hours = 78 bars).
        trend: 'up', 'down', or 'flat'.
    """
    rng = np.random.RandomState(42)
    total = n_days * bars_per_day
    dates = pd.date_range("2023-01-02 09:30", periods=total, freq="5min")

    price = 100.0
    rows = []
    for i in range(total):
        if trend == "up":
            drift = 0.00015
        elif trend == "down":
            drift = -0.00015
        else:
            drift = 0.0

        ret = drift + rng.randn() * 0.001
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.001)
        low = price * (1 - abs(rng.randn()) * 0.001)
        rows.append({
            "open": price * (1 + rng.randn() * 0.0005),
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(10_000, 100_000)),
        })

    return pd.DataFrame(rows, index=dates)


def _make_daily_df(n_bars=100, trend="up"):
    """Create daily OHLCV for direct HTF usage."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2022-06-01", periods=n_bars)

    price = 100.0
    rows = []
    for i in range(n_bars):
        if trend == "up":
            drift = 0.003
        elif trend == "down":
            drift = -0.003
        else:
            drift = 0.0

        ret = drift + rng.randn() * 0.01
        price *= 1 + ret
        high = price * 1.005
        low = price * 0.995
        rows.append({
            "open": price * 0.999,
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })

    return pd.DataFrame(rows, index=dates)


# ═══════════════════════════════════════════════════════════════════════
#  1. Resample to Daily
# ═══════════════════════════════════════════════════════════════════════


class TestResampleToDaily:

    def test_resample_to_daily_reduces_rows(self, mtf):
        df = _make_intraday_df(n_days=60, bars_per_day=78)
        htf = mtf.resample_to_htf(df, period="1D")
        # Resampling intraday to daily should always reduce row count
        assert len(htf) < len(df)
        assert len(htf) >= 10

    def test_resample_preserves_ohlcv_columns(self, mtf):
        df = _make_intraday_df(n_days=30)
        htf = mtf.resample_to_htf(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in htf.columns

    def test_resample_high_is_max(self, mtf):
        df = _make_intraday_df(n_days=10, bars_per_day=10)
        htf = mtf.resample_to_htf(df)
        # For each day, the HTF high should be >= all intraday highs
        for idx in htf.index:
            day_str = str(idx.date())
            day_mask = df.index.strftime("%Y-%m-%d") == day_str
            if day_mask.sum() > 0:
                assert htf.loc[idx, "high"] >= df.loc[day_mask, "high"].max() - 0.01

    def test_resample_volume_is_sum(self, mtf):
        df = _make_intraday_df(n_days=5, bars_per_day=10)
        htf = mtf.resample_to_htf(df)
        for idx in htf.index:
            day_str = str(idx.date())
            day_mask = df.index.strftime("%Y-%m-%d") == day_str
            if day_mask.sum() > 0:
                assert htf.loc[idx, "volume"] == df.loc[day_mask, "volume"].sum()


# ═══════════════════════════════════════════════════════════════════════
#  2. Bullish Trend
# ═══════════════════════════════════════════════════════════════════════


class TestBullishTrend:

    def test_uptrend_returns_bullish(self, mtf):
        df = _make_daily_df(n_bars=100, trend="up")
        trend = mtf.get_htf_trend(df)
        assert trend == "BULLISH"

    def test_strong_uptrend(self):
        mtf = MultiTimeframeTrend(sma_fast=10, sma_slow=30)
        df = _make_daily_df(n_bars=60, trend="up")
        assert mtf.get_htf_trend(df) == "BULLISH"


# ═══════════════════════════════════════════════════════════════════════
#  3. Bearish Trend
# ═══════════════════════════════════════════════════════════════════════


class TestBearishTrend:

    def test_downtrend_returns_bearish(self, mtf):
        df = _make_daily_df(n_bars=100, trend="down")
        trend = mtf.get_htf_trend(df)
        assert trend == "BEARISH"


# ═══════════════════════════════════════════════════════════════════════
#  4. Neutral Trend
# ═══════════════════════════════════════════════════════════════════════


class TestNeutralTrend:

    def test_sideways_returns_neutral(self, mtf):
        df = _make_daily_df(n_bars=100, trend="flat")
        trend = mtf.get_htf_trend(df)
        assert trend == "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
#  5. Alignment Checks
# ═══════════════════════════════════════════════════════════════════════


class TestAlignment:

    def test_buy_aligned_in_bullish(self, mtf):
        df = _make_daily_df(n_bars=100, trend="up")
        assert mtf.is_aligned(df, "BUY") is True

    def test_buy_not_aligned_in_bearish(self, mtf):
        df = _make_daily_df(n_bars=100, trend="down")
        assert mtf.is_aligned(df, "BUY") is False

    def test_sell_aligned_in_bearish(self, mtf):
        df = _make_daily_df(n_bars=100, trend="down")
        assert mtf.is_aligned(df, "SELL") is True

    def test_sell_not_aligned_in_bullish(self, mtf):
        df = _make_daily_df(n_bars=100, trend="up")
        assert mtf.is_aligned(df, "SELL") is False

    def test_buy_aligned_in_neutral(self, mtf):
        df = _make_daily_df(n_bars=100, trend="flat")
        assert mtf.is_aligned(df, "BUY") is True

    def test_sell_aligned_in_neutral(self, mtf):
        df = _make_daily_df(n_bars=100, trend="flat")
        assert mtf.is_aligned(df, "SELL") is True

    def test_unknown_direction_always_aligned(self, mtf):
        df = _make_daily_df(n_bars=100, trend="up")
        assert mtf.is_aligned(df, "HOLD") is True


# ═══════════════════════════════════════════════════════════════════════
#  6. Support / Resistance
# ═══════════════════════════════════════════════════════════════════════


class TestSupportResistance:

    def test_returns_correct_keys(self, mtf):
        df = _make_daily_df(n_bars=60)
        sr = mtf.get_htf_support_resistance(df)
        assert "support" in sr
        assert "resistance" in sr
        assert "pivot" in sr

    def test_support_below_resistance(self, mtf):
        df = _make_daily_df(n_bars=60)
        sr = mtf.get_htf_support_resistance(df)
        assert sr["support"] < sr["resistance"]

    def test_pivot_between_support_and_resistance(self, mtf):
        df = _make_daily_df(n_bars=60)
        sr = mtf.get_htf_support_resistance(df)
        assert sr["support"] <= sr["pivot"] <= sr["resistance"]


# ═══════════════════════════════════════════════════════════════════════
#  7. Insufficient Data
# ═══════════════════════════════════════════════════════════════════════


class TestInsufficientData:

    def test_short_data_returns_neutral(self, mtf):
        df = _make_daily_df(n_bars=10)
        trend = mtf.get_htf_trend(df)
        assert trend == "NEUTRAL"

    def test_very_short_data(self, mtf):
        df = _make_daily_df(n_bars=3)
        trend = mtf.get_htf_trend(df)
        assert trend == "NEUTRAL"

    def test_insufficient_sr_returns_defaults(self, mtf):
        df = _make_daily_df(n_bars=3)
        sr = mtf.get_htf_support_resistance(df, lookback=20)
        assert sr["support"] == 0
        assert sr["resistance"] == float("inf")


# ═══════════════════════════════════════════════════════════════════════
#  8. Non-Datetime Index
# ═══════════════════════════════════════════════════════════════════════


class TestNonDatetimeIndex:

    def test_non_datetime_index_returns_original(self, mtf):
        """resample_to_htf should return the original df if index is not DatetimeIndex."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "open": rng.uniform(100, 110, 50),
            "high": rng.uniform(110, 120, 50),
            "low": rng.uniform(90, 100, 50),
            "close": rng.uniform(100, 110, 50),
            "volume": rng.randint(1000, 10000, 50),
        })
        result = mtf.resample_to_htf(df)
        assert len(result) == len(df)

    def test_non_datetime_trend_neutral_or_computed(self, mtf):
        """With non-datetime index, resample returns same df, so trend still computed."""
        rng = np.random.RandomState(42)
        n = 100
        prices = np.cumsum(rng.randn(n) * 0.5) + 100
        df = pd.DataFrame({
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": rng.randint(1000, 10000, n),
        })
        trend = mtf.get_htf_trend(df)
        assert trend in ("BULLISH", "BEARISH", "NEUTRAL")
