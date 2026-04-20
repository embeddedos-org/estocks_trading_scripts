"""Tests for the Squeeze Momentum indicator.

Covers squeeze-on detection (low volatility), squeeze-off detection
(volatility expansion), momentum sign, output column completeness,
and edge cases (empty data, short data).

10+ tests total.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.indicators.technical_indicators import TechnicalIndicators as TI


def _make_ohlcv(n=200, volatility="normal"):
    """Create synthetic OHLCV data with controlled volatility.

    Args:
        n: Number of bars.
        volatility: 'low' for compression, 'high' for expansion, 'normal'.
    """
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    price = 100.0
    rows = []

    for i in range(n):
        if volatility == "low":
            noise = rng.randn() * 0.001
            spread = 0.001
        elif volatility == "high":
            noise = rng.randn() * 0.03
            spread = 0.02
        else:
            noise = rng.randn() * 0.01
            spread = 0.005

        price *= 1 + noise
        rows.append({
            "open": price * (1 + rng.randn() * 0.001),
            "high": price * (1 + spread),
            "low": price * (1 - spread),
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })

    return pd.DataFrame(rows, index=dates)


@pytest.fixture
def normal_df():
    return _make_ohlcv(200, "normal")


@pytest.fixture
def low_vol_df():
    return _make_ohlcv(200, "low")


@pytest.fixture
def high_vol_df():
    return _make_ohlcv(200, "high")


# ═══════════════════════════════════════════════════════════════════════
#  1. Squeeze On (Low Volatility)
# ═══════════════════════════════════════════════════════════════════════


class TestSqueezeOnLowVol:

    def test_low_vol_produces_squeeze_on(self, low_vol_df):
        result = TI.squeeze(low_vol_df)
        # Low vol → BB should be inside KC → squeeze_on should be True
        squeeze_on_after_warmup = result["squeeze_on"].dropna().tail(100)
        on_count = squeeze_on_after_warmup.sum()
        assert on_count > 20, (
            f"Expected significant squeeze_on=True for low vol, got {on_count}/100"
        )

    def test_squeeze_on_is_bool(self, low_vol_df):
        result = TI.squeeze(low_vol_df)
        valid = result["squeeze_on"].dropna()
        assert valid.dtype == bool or all(v in (True, False) for v in valid)


# ═══════════════════════════════════════════════════════════════════════
#  2. Squeeze Off (Expansion)
# ═══════════════════════════════════════════════════════════════════════


class TestSqueezeOffExpansion:

    def test_high_vol_produces_squeeze_off(self, high_vol_df):
        result = TI.squeeze(high_vol_df)
        squeeze_off_after_warmup = result["squeeze_off"].dropna().tail(100)
        off_count = squeeze_off_after_warmup.sum()
        assert off_count > 50, (
            f"Expected majority squeeze_off=True for high vol, got {off_count}/100"
        )

    def test_squeeze_off_is_complement(self, normal_df):
        result = TI.squeeze(normal_df)
        valid = result.dropna()
        if len(valid) > 0:
            assert (valid["squeeze_on"] | valid["squeeze_off"]).all()
            assert not (valid["squeeze_on"] & valid["squeeze_off"]).any()


# ═══════════════════════════════════════════════════════════════════════
#  3. Momentum
# ═══════════════════════════════════════════════════════════════════════


class TestMomentum:

    def test_momentum_positive_in_uptrend(self):
        """Uptrend data should show positive momentum in later bars."""
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        price = 100.0
        rows = []
        for _ in range(n):
            price *= 1.003 + rng.randn() * 0.002
            rows.append({
                "open": price * 0.999,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": 1_000_000,
            })
        df = pd.DataFrame(rows, index=dates)
        result = TI.squeeze(df)
        last_momentum = result["momentum"].dropna().tail(20).mean()
        assert last_momentum > 0, "Uptrend should produce positive momentum"

    def test_momentum_is_numeric(self, normal_df):
        result = TI.squeeze(normal_df)
        assert pd.api.types.is_numeric_dtype(result["momentum"])


# ═══════════════════════════════════════════════════════════════════════
#  4. Columns Present
# ═══════════════════════════════════════════════════════════════════════


class TestColumnsPresent:

    def test_all_expected_columns(self, normal_df):
        result = TI.squeeze(normal_df)
        expected = {"squeeze_on", "squeeze_off", "momentum",
                    "bb_upper", "bb_lower", "kc_upper", "kc_lower"}
        assert set(result.columns) == expected

    def test_index_preserved(self, normal_df):
        result = TI.squeeze(normal_df)
        assert len(result) == len(normal_df)
        assert (result.index == normal_df.index).all()

    def test_bb_upper_above_bb_lower(self, normal_df):
        result = TI.squeeze(normal_df)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_kc_upper_above_kc_lower(self, normal_df):
        result = TI.squeeze(normal_df)
        valid = result.dropna()
        assert (valid["kc_upper"] >= valid["kc_lower"]).all()


# ═══════════════════════════════════════════════════════════════════════
#  5. Edge Cases — Empty and Short Data
# ═══════════════════════════════════════════════════════════════════════


class TestSqueezeEdgeCases:

    def test_empty_data(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = TI.squeeze(df)
        assert len(result) == 0
        assert set(result.columns) == {
            "squeeze_on", "squeeze_off", "momentum",
            "bb_upper", "bb_lower", "kc_upper", "kc_lower",
        }

    def test_short_data_returns_nans(self):
        """With fewer bars than the lookback, result should be mostly NaN."""
        df = _make_ohlcv(5)
        result = TI.squeeze(df, bb_length=20, kc_length=20)
        assert len(result) == 5
        # All should be NaN since we have < 20 bars
        assert result["bb_upper"].isna().all()

    def test_custom_parameters(self, normal_df):
        result = TI.squeeze(
            normal_df, bb_length=10, bb_mult=1.5, kc_length=10, kc_mult=1.0
        )
        assert len(result) == len(normal_df)
        assert result["bb_upper"].dropna().shape[0] > 0
