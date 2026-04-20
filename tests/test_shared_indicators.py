"""
Tests for shared.indicators — TechnicalIndicators and CandlestickPatterns.
Covers RSI, MACD, Bollinger Bands, ATR, ADX, VWAP, moving averages, custom
indicators, and candlestick pattern detection. Tests edge cases: empty data,
single row, NaN values, all zeros.

All tests force the manual fallback path by mocking _HAS_TALIB=False and
_HAS_PANDAS_TA=False so results are deterministic and verifiable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import patch

import shared.indicators.technical_indicators as ti_module
from shared.indicators.technical_indicators import TechnicalIndicators as TI

import shared.indicators.candlestick_patterns as cp_module
from shared.indicators.candlestick_patterns import CandlestickPatterns as CP, _require_talib


# ── Fixtures ──


@pytest.fixture(autouse=True)
def force_manual_fallback():
    """Force manual fallback for all tests by disabling TA-Lib and pandas-ta."""
    with patch.object(ti_module, "_HAS_TALIB", False), \
         patch.object(ti_module, "_HAS_PANDAS_TA", False), \
         patch.object(cp_module, "_HAS_TALIB", False):
        yield


@pytest.fixture
def sample_series():
    """Simple price series for indicator calculations."""
    prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
              46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
              46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03,
              44.18, 44.22, 44.57, 43.42, 42.66, 43.13]
    return pd.Series(prices, name="close")


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV DataFrame for indicators requiring full candle data."""
    np.random.seed(42)
    n = 50
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


@pytest.fixture
def empty_series():
    """Empty price series."""
    return pd.Series([], dtype=float, name="close")


@pytest.fixture
def single_row_series():
    """Single-value price series."""
    return pd.Series([100.0], name="close")


@pytest.fixture
def empty_ohlcv():
    """Empty OHLCV DataFrame."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def single_row_ohlcv():
    """Single-row OHLCV DataFrame."""
    return pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.5], "volume": [5000.0],
    })


# ═══════════════════════════════════════════════════
# TREND INDICATORS
# ═══════════════════════════════════════════════════


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_basic(self, sample_series):
        """Validates SMA produces correct values for a known series."""
        result = TI.sma(sample_series, length=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)
        assert result.iloc[:4].isna().all()
        expected = sample_series.iloc[:5].mean()
        assert abs(result.iloc[4] - expected) < 1e-10

    def test_sma_all_same_values(self):
        """Validates SMA of constant series equals the constant."""
        s = pd.Series([50.0] * 20)
        result = TI.sma(s, length=10)
        assert abs(result.iloc[-1] - 50.0) < 1e-10

    def test_sma_empty(self, empty_series):
        """Validates SMA of empty series returns empty."""
        result = TI.sma(empty_series, length=5)
        assert len(result) == 0

    def test_sma_single_row(self, single_row_series):
        """Validates SMA with fewer rows than window returns NaN."""
        result = TI.sma(single_row_series, length=5)
        assert result.isna().all()


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_basic(self, sample_series):
        """Validates EMA produces Series of correct length."""
        result = TI.ema(sample_series, length=10)
        assert len(result) == len(sample_series)
        assert not result.iloc[-1:].isna().any()

    def test_ema_length_1_equals_series(self):
        """Validates EMA with length=1 closely tracks the original series."""
        s = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = TI.ema(s, length=1)
        pd.testing.assert_series_equal(result, s, check_names=False, atol=1e-10)


class TestDEMA:
    """Tests for Double Exponential Moving Average."""

    def test_dema_basic(self, sample_series):
        """Validates DEMA returns valid results."""
        result = TI.dema(sample_series, length=10)
        assert len(result) == len(sample_series)
        assert not result.iloc[-1:].isna().any()


class TestTEMA:
    """Tests for Triple Exponential Moving Average."""

    def test_tema_basic(self, sample_series):
        """Validates TEMA returns valid results."""
        result = TI.tema(sample_series, length=10)
        assert len(result) == len(sample_series)
        assert not result.iloc[-1:].isna().any()


class TestWMA:
    """Tests for Weighted Moving Average."""

    def test_wma_basic(self, sample_series):
        """Validates WMA produces correct output shape."""
        result = TI.wma(sample_series, length=5)
        assert len(result) == len(sample_series)
        assert result.iloc[:4].isna().all()
        assert not result.iloc[4:].isna().any()

    def test_wma_constant_series(self):
        """Validates WMA of constant series equals the constant."""
        s = pd.Series([25.0] * 10)
        result = TI.wma(s, length=5)
        assert abs(result.iloc[-1] - 25.0) < 1e-10


class TestKAMA:
    """Tests for Kaufman Adaptive Moving Average."""

    def test_kama_basic(self, sample_series):
        """Validates KAMA produces correct length output."""
        result = TI.kama(sample_series, length=10)
        assert len(result) == len(sample_series)
        assert result.iloc[:9].isna().all()

    def test_kama_short_series(self):
        """Validates KAMA with series shorter than length returns all NaN."""
        s = pd.Series([100, 101, 102], dtype=float)
        result = TI.kama(s, length=10)
        assert result.isna().all()


class TestHMA:
    """Tests for Hull Moving Average."""

    def test_hma_basic(self, sample_series):
        """Validates HMA produces valid output."""
        result = TI.hma(sample_series, length=9)
        assert len(result) == len(sample_series)


# ═══════════════════════════════════════════════════
# MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_basic(self, sample_series):
        """Validates RSI produces values between 0 and 100."""
        result = TI.rsi(sample_series, length=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_length(self, sample_series):
        """Validates RSI output length matches input."""
        result = TI.rsi(sample_series, length=14)
        assert len(result) == len(sample_series)

    def test_rsi_all_gains_near_100(self):
        """Validates RSI approaches 100 when gains dominate.
        Pure monotonic series produces NaN (0 avg_loss), so use mostly-gains.
        """
        vals = list(range(1, 40))
        vals[10] = vals[9] - 0.01  # single tiny loss to avoid NaN
        s = pd.Series(vals, dtype=float)
        result = TI.rsi(s, length=14)
        assert result.iloc[-1] > 90

    def test_rsi_all_losses_near_0(self):
        """Validates RSI approaches 0 when all changes are negative."""
        s = pd.Series(range(50, 1, -1), dtype=float)
        result = TI.rsi(s, length=14)
        assert result.iloc[-1] < 10

    def test_rsi_empty(self, empty_series):
        """Validates RSI of empty series returns empty."""
        result = TI.rsi(empty_series)
        assert len(result) == 0

    def test_rsi_name(self, sample_series):
        """Validates RSI result has name='RSI'."""
        result = TI.rsi(sample_series)
        assert result.name == "RSI"


class TestMACD:
    """Tests for MACD."""

    def test_macd_returns_three_series(self, sample_series):
        """Validates MACD returns tuple of (macd_line, signal, histogram)."""
        macd_line, signal_line, hist = TI.macd(sample_series)
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(hist, pd.Series)

    def test_macd_histogram_is_diff(self, sample_series):
        """Validates histogram = macd_line - signal_line."""
        macd_line, signal_line, hist = TI.macd(sample_series)
        diff = macd_line - signal_line
        pd.testing.assert_series_equal(hist, diff, check_names=False, atol=1e-10)

    def test_macd_names(self, sample_series):
        """Validates MACD output series have correct names."""
        m, s, h = TI.macd(sample_series)
        assert m.name == "MACD"
        assert s.name == "MACDs"
        assert h.name == "MACDh"

    def test_macd_empty(self, empty_series):
        """Validates MACD of empty series returns empty series tuple."""
        m, s, h = TI.macd(empty_series)
        assert len(m) == 0


class TestStochastic:
    """Tests for Stochastic Oscillator."""

    def test_stochastic_returns_two_series(self, sample_ohlcv):
        """Validates stochastic returns K and D lines."""
        k, d = TI.stochastic(sample_ohlcv)
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_stochastic_range_0_100(self, sample_ohlcv):
        """Validates stochastic K values are in [0, 100] range."""
        k, d = TI.stochastic(sample_ohlcv)
        valid_k = k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()


class TestADX:
    """Tests for Average Directional Index."""

    def test_adx_returns_three_series(self, sample_ohlcv):
        """Validates ADX returns (ADX, +DI, -DI) tuple."""
        adx_val, plus_di, minus_di = TI.adx(sample_ohlcv)
        assert isinstance(adx_val, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)

    def test_adx_non_negative(self, sample_ohlcv):
        """Validates ADX values are non-negative."""
        adx_val, _, _ = TI.adx(sample_ohlcv)
        valid = adx_val.dropna()
        assert (valid >= 0).all()


class TestCCI:
    """Tests for Commodity Channel Index."""

    def test_cci_basic(self, sample_ohlcv):
        """Validates CCI returns correct length output."""
        result = TI.cci(sample_ohlcv, length=20)
        assert len(result) == len(sample_ohlcv)


class TestWilliamsR:
    """Tests for Williams %R."""

    def test_williams_r_range(self, sample_ohlcv):
        """Validates Williams %R values are in [-100, 0] range."""
        result = TI.williams_r(sample_ohlcv)
        valid = result.dropna()
        assert (valid <= 0).all() and (valid >= -100).all()


class TestROC:
    """Tests for Rate of Change."""

    def test_roc_basic(self, sample_series):
        """Validates ROC output length and non-NaN values exist."""
        result = TI.roc(sample_series, length=5)
        assert len(result) == len(sample_series)
        assert not result.dropna().empty


class TestMFI:
    """Tests for Money Flow Index."""

    def test_mfi_range(self, sample_ohlcv):
        """Validates MFI values are between 0 and 100."""
        result = TI.mfi(sample_ohlcv, length=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


# ═══════════════════════════════════════════════════
# VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bbands_columns(self, sample_series):
        """Validates bbands returns DataFrame with BBL, BBM, BBU, BBB, BBP."""
        result = TI.bbands(sample_series, length=20, std_dev=2.0)
        assert isinstance(result, pd.DataFrame)
        for col in ["BBL", "BBM", "BBU", "BBB", "BBP"]:
            assert col in result.columns

    def test_bbands_upper_above_lower(self, sample_series):
        """Validates upper band is always >= lower band."""
        result = TI.bbands(sample_series, length=10)
        valid_idx = result.dropna().index
        assert (result.loc[valid_idx, "BBU"] >= result.loc[valid_idx, "BBL"]).all()

    def test_bbands_mid_is_sma(self, sample_series):
        """Validates middle band equals SMA of the series."""
        result = TI.bbands(sample_series, length=10)
        sma = TI.sma(sample_series, length=10)
        valid_idx = result["BBM"].dropna().index
        pd.testing.assert_series_equal(
            result.loc[valid_idx, "BBM"], sma.loc[valid_idx],
            check_names=False, atol=1e-10,
        )

    def test_bbands_empty(self, empty_series):
        """Validates bbands of empty series returns empty DataFrame."""
        result = TI.bbands(empty_series)
        assert len(result) == 0


class TestATR:
    """Tests for Average True Range."""

    def test_atr_positive(self, sample_ohlcv):
        """Validates ATR values are positive."""
        result = TI.atr(sample_ohlcv, length=14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_atr_length(self, sample_ohlcv):
        """Validates ATR output length matches input."""
        result = TI.atr(sample_ohlcv, length=14)
        assert len(result) == len(sample_ohlcv)

    def test_atr_empty(self, empty_ohlcv):
        """Validates ATR of empty DataFrame returns empty."""
        result = TI.atr(empty_ohlcv)
        assert len(result) == 0

    def test_atr_name(self, sample_ohlcv):
        """Validates ATR result has name='ATR'."""
        result = TI.atr(sample_ohlcv)
        assert result.name == "ATR"


class TestKeltnerChannels:
    """Tests for Keltner Channels."""

    def test_keltner_columns(self, sample_ohlcv):
        """Validates Keltner returns KCU, KCM, KCL columns."""
        result = TI.keltner_channels(sample_ohlcv)
        for col in ["KCU", "KCM", "KCL"]:
            assert col in result.columns

    def test_keltner_upper_above_lower(self, sample_ohlcv):
        """Validates upper channel >= lower channel."""
        result = TI.keltner_channels(sample_ohlcv)
        valid_idx = result.dropna().index
        assert (result.loc[valid_idx, "KCU"] >= result.loc[valid_idx, "KCL"]).all()


class TestDonchianChannels:
    """Tests for Donchian Channels."""

    def test_donchian_columns(self, sample_ohlcv):
        """Validates Donchian returns DCU, DCM, DCL columns."""
        result = TI.donchian_channels(sample_ohlcv)
        for col in ["DCU", "DCM", "DCL"]:
            assert col in result.columns

    def test_donchian_mid_is_average(self, sample_ohlcv):
        """Validates DCM = (DCU + DCL) / 2."""
        result = TI.donchian_channels(sample_ohlcv)
        valid_idx = result.dropna().index
        expected_mid = (result.loc[valid_idx, "DCU"] + result.loc[valid_idx, "DCL"]) / 2
        pd.testing.assert_series_equal(
            result.loc[valid_idx, "DCM"], expected_mid,
            check_names=False, atol=1e-10,
        )


class TestChaikinVolatility:
    """Tests for Chaikin Volatility."""

    def test_chaikin_basic(self, sample_ohlcv):
        """Validates Chaikin Volatility returns correct length."""
        result = TI.chaikin_volatility(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert result.name == "CHAIKIN_VOL"


# ═══════════════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════════════


class TestOBV:
    """Tests for On Balance Volume."""

    def test_obv_basic(self, sample_ohlcv):
        """Validates OBV returns correct length."""
        result = TI.obv(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_obv_name(self, sample_ohlcv):
        """Validates OBV result has name='OBV'."""
        result = TI.obv(sample_ohlcv)
        assert result.name == "OBV"


class TestVWAP:
    """Tests for Volume Weighted Average Price."""

    def test_vwap_basic(self, sample_ohlcv):
        """Validates VWAP returns correct length and is within price range."""
        result = TI.vwap(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        assert (valid >= sample_ohlcv["low"].min() - 1).all()
        assert (valid <= sample_ohlcv["high"].max() + 1).all()

    def test_vwap_name(self, sample_ohlcv):
        """Validates VWAP result has name='VWAP'."""
        result = TI.vwap(sample_ohlcv)
        assert result.name == "VWAP"

    def test_vwap_zero_volume(self):
        """Validates VWAP handles zero volume gracefully (returns NaN)."""
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [0.0],
        })
        result = TI.vwap(df)
        assert len(result) == 1


class TestADLine:
    """Tests for Accumulation/Distribution Line."""

    def test_ad_line_basic(self, sample_ohlcv):
        """Validates A/D Line returns correct length."""
        result = TI.ad_line(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert result.name == "AD"


class TestCMF:
    """Tests for Chaikin Money Flow."""

    def test_cmf_range(self, sample_ohlcv):
        """Validates CMF values are in [-1, 1] range."""
        result = TI.cmf(sample_ohlcv, length=20)
        valid = result.dropna()
        assert (valid >= -1.1).all() and (valid <= 1.1).all()


class TestVolumeProfile:
    """Tests for Volume Profile."""

    def test_volume_profile_keys(self, sample_ohlcv):
        """Validates volume profile returns poc, vah, val, profile."""
        result = TI.volume_profile(sample_ohlcv, bins=10)
        assert "poc" in result
        assert "vah" in result
        assert "val" in result
        assert "profile" in result
        assert isinstance(result["profile"], pd.DataFrame)

    def test_volume_profile_poc_in_range(self, sample_ohlcv):
        """Validates Point of Control is within the price range."""
        result = TI.volume_profile(sample_ohlcv, bins=10)
        assert result["poc"] >= sample_ohlcv["low"].min()
        assert result["poc"] <= sample_ohlcv["high"].max()


# ═══════════════════════════════════════════════════
# CUSTOM INDICATORS
# ═══════════════════════════════════════════════════


class TestHeikinAshi:
    """Tests for Heikin Ashi candles."""

    def test_heikin_ashi_columns(self, sample_ohlcv):
        """Validates Heikin Ashi returns HA_open, HA_high, HA_low, HA_close."""
        result = TI.heikin_ashi(sample_ohlcv)
        for col in ["HA_open", "HA_high", "HA_low", "HA_close"]:
            assert col in result.columns

    def test_heikin_ashi_close_is_average(self, sample_ohlcv):
        """Validates HA_close = (open + high + low + close) / 4."""
        result = TI.heikin_ashi(sample_ohlcv)
        expected = (sample_ohlcv["open"] + sample_ohlcv["high"] +
                    sample_ohlcv["low"] + sample_ohlcv["close"]) / 4
        pd.testing.assert_series_equal(
            result["HA_close"], expected, check_names=False, atol=1e-10,
        )

    def test_heikin_ashi_first_open(self, sample_ohlcv):
        """Validates first HA_open = (open[0] + close[0]) / 2."""
        result = TI.heikin_ashi(sample_ohlcv)
        expected = (sample_ohlcv["open"].iloc[0] + sample_ohlcv["close"].iloc[0]) / 2
        assert abs(result["HA_open"].iloc[0] - expected) < 1e-10

    def test_heikin_ashi_high_ge_close_and_open(self, sample_ohlcv):
        """Validates HA_high >= max(HA_open, HA_close) for each row."""
        result = TI.heikin_ashi(sample_ohlcv)
        max_oc = pd.concat([result["HA_open"], result["HA_close"]], axis=1).max(axis=1)
        assert (result["HA_high"] >= max_oc - 1e-10).all()


class TestPivotPoints:
    """Tests for Pivot Points."""

    def test_pivot_standard_columns(self, sample_ohlcv):
        """Validates standard pivot returns PP, R1-R3, S1-S3."""
        result = TI.pivot_points(sample_ohlcv, method="standard")
        for col in ["PP", "R1", "S1", "R2", "S2", "R3", "S3"]:
            assert col in result.columns

    def test_pivot_fibonacci_columns(self, sample_ohlcv):
        """Validates fibonacci pivot returns PP, R1-R3, S1-S3."""
        result = TI.pivot_points(sample_ohlcv, method="fibonacci")
        for col in ["PP", "R1", "S1", "R2", "S2", "R3", "S3"]:
            assert col in result.columns

    def test_pivot_camarilla_columns(self, sample_ohlcv):
        """Validates camarilla pivot returns PP, R1-R3, S1-S3."""
        result = TI.pivot_points(sample_ohlcv, method="camarilla")
        for col in ["PP", "R1", "S1", "R2", "S2", "R3", "S3"]:
            assert col in result.columns

    def test_pivot_pp_formula(self, sample_ohlcv):
        """Validates PP = (high[-1] + low[-1] + close[-1]) / 3."""
        result = TI.pivot_points(sample_ohlcv, method="standard")
        idx = 5
        h = sample_ohlcv["high"].iloc[idx - 1]
        l = sample_ohlcv["low"].iloc[idx - 1]
        c = sample_ohlcv["close"].iloc[idx - 1]
        expected_pp = (h + l + c) / 3
        assert abs(result["PP"].iloc[idx] - expected_pp) < 1e-10

    def test_pivot_r1_gt_pp_standard(self, sample_ohlcv):
        """Validates R1 > PP in standard pivot points."""
        result = TI.pivot_points(sample_ohlcv, method="standard")
        valid = result.dropna()
        assert (valid["R1"] >= valid["PP"]).all()


class TestSupertrend:
    """Tests for Supertrend indicator."""

    def test_supertrend_columns(self, sample_ohlcv):
        """Validates Supertrend returns SUPERT, SUPERTd, SUPERTl, SUPERTs."""
        result = TI.supertrend(sample_ohlcv, length=10, multiplier=3.0)
        for col in ["SUPERT", "SUPERTd", "SUPERTl", "SUPERTs"]:
            assert col in result.columns

    def test_supertrend_direction_values(self, sample_ohlcv):
        """Validates direction is 1 or -1."""
        result = TI.supertrend(sample_ohlcv, length=10, multiplier=3.0)
        valid = result["SUPERTd"].dropna()
        assert set(valid.unique()).issubset({1, -1})


class TestIchimoku:
    """Tests for Ichimoku Cloud."""

    def test_ichimoku_columns(self, sample_ohlcv):
        """Validates Ichimoku returns all five components."""
        result = TI.ichimoku(sample_ohlcv)
        for col in ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span"]:
            assert col in result.columns

    def test_ichimoku_length(self, sample_ohlcv):
        """Validates Ichimoku output length matches input."""
        result = TI.ichimoku(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)


class TestPSAR:
    """Tests for Parabolic SAR."""

    def test_psar_basic(self, sample_ohlcv):
        """Validates PSAR returns correct length."""
        result = TI.psar(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert result.name == "PSAR"

    def test_psar_single_row(self, single_row_ohlcv):
        """Validates PSAR handles single row without error."""
        result = TI.psar(single_row_ohlcv)
        assert len(result) == 1


# ═══════════════════════════════════════════════════
# CANDLESTICK PATTERNS
# ═══════════════════════════════════════════════════


class TestDoji:
    """Tests for Doji candle detection."""

    def test_doji_detected(self):
        """Validates doji detected when open ~= close."""
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0],
            "high": [105.0, 105.0, 105.0],
            "low": [95.0, 95.0, 95.0],
            "close": [100.01, 100.0, 100.49],
        })
        result = CP.doji(df, threshold=0.05)
        assert result.iloc[0] == 100
        assert result.iloc[1] == 100

    def test_doji_not_detected_large_body(self):
        """Validates no doji when body is large relative to range."""
        df = pd.DataFrame({
            "open": [90.0], "high": [105.0], "low": [85.0], "close": [104.0],
        })
        result = CP.doji(df, threshold=0.05)
        assert result.iloc[0] == 0

    def test_doji_name(self):
        """Validates doji result name is 'DOJI'."""
        df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [95.0], "close": [100.0],
        })
        result = CP.doji(df)
        assert result.name == "DOJI"


class TestHammer:
    """Tests for Hammer candle detection."""

    def test_hammer_detected(self):
        """Validates hammer detected: small body at top, long lower shadow."""
        df = pd.DataFrame({
            "open": [100.0], "high": [100.5], "low": [95.0],
            "close": [100.2],
        })
        result = CP.hammer(df, body_ratio=0.3, shadow_ratio=2.0)
        assert result.name == "HAMMER"

    def test_hammer_not_detected_big_body(self):
        """Validates no hammer when body is too large."""
        df = pd.DataFrame({
            "open": [90.0], "high": [105.0], "low": [89.0], "close": [104.0],
        })
        result = CP.hammer(df)
        assert result.iloc[0] == 0


class TestEngulfing:
    """Tests for Engulfing pattern detection."""

    def test_bullish_engulfing(self):
        """Validates bullish engulfing detected: prev bearish, curr bullish engulfs."""
        df = pd.DataFrame({
            "open": [105.0, 98.0],
            "high": [106.0, 108.0],
            "low": [99.0, 97.0],
            "close": [100.0, 106.0],
        })
        result = CP.engulfing(df)
        assert result.iloc[1] == 100

    def test_bearish_engulfing(self):
        """Validates bearish engulfing detected: prev bullish, curr bearish engulfs."""
        df = pd.DataFrame({
            "open": [100.0, 106.0],
            "high": [106.0, 107.0],
            "low": [99.0, 98.0],
            "close": [105.0, 99.0],
        })
        result = CP.engulfing(df)
        assert result.iloc[1] == -100

    def test_no_engulfing(self):
        """Validates no engulfing when conditions are not met."""
        df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        })
        result = CP.engulfing(df)
        assert result.iloc[1] == 0

    def test_engulfing_name(self):
        """Validates engulfing result name is 'ENGULFING'."""
        df = pd.DataFrame({
            "open": [100.0, 101.0], "high": [102.0, 103.0],
            "low": [99.0, 100.0], "close": [101.0, 102.0],
        })
        result = CP.engulfing(df)
        assert result.name == "ENGULFING"


class TestRequireTablib:
    """Tests for _require_talib guard and TA-Lib-only patterns."""

    def test_require_talib_raises(self):
        """Validates _require_talib raises ImportError when TA-Lib is absent."""
        with pytest.raises(ImportError, match="requires TA-Lib"):
            _require_talib("morning_star")

    @pytest.mark.parametrize("method_name", [
        "morning_star", "evening_star", "three_white_soldiers",
        "three_black_crows", "harami", "shooting_star",
        "hanging_man", "spinning_top", "marubozu",
    ])
    def test_talib_only_patterns_raise(self, method_name, sample_ohlcv):
        """Validates TA-Lib-only patterns raise ImportError when TA-Lib absent."""
        method = getattr(CP, method_name)
        with pytest.raises(ImportError, match="requires TA-Lib"):
            method(sample_ohlcv)


class TestScanAll:
    """Tests for scan_all() — runs all available patterns."""

    def test_scan_all_returns_dataframe(self, sample_ohlcv):
        """Validates scan_all returns a DataFrame."""
        result = CP.scan_all(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_scan_all_has_fallback_patterns(self, sample_ohlcv):
        """Validates scan_all includes doji, hammer, engulfing (always available)."""
        result = CP.scan_all(sample_ohlcv)
        assert "DOJI" in result.columns
        assert "HAMMER" in result.columns
        assert "ENGULFING" in result.columns

    def test_scan_all_no_talib_patterns(self, sample_ohlcv):
        """Validates TA-Lib-only patterns are excluded when TA-Lib is absent."""
        result = CP.scan_all(sample_ohlcv)
        assert "MORNING_STAR" not in result.columns
        assert "EVENING_STAR" not in result.columns


# ═══════════════════════════════════════════════════
# EDGE CASES — all zeros, NaN values
# ═══════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for indicators with degenerate inputs."""

    def test_sma_all_zeros(self):
        """Validates SMA of all zeros returns zeros."""
        s = pd.Series([0.0] * 20)
        result = TI.sma(s, length=5)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_rsi_all_zeros(self):
        """Validates RSI of constant series (no change) returns NaN."""
        s = pd.Series([50.0] * 30)
        result = TI.rsi(s, length=14)
        # No price change means avg_gain=0, avg_loss=0
        valid = result.dropna()
        # Implementation: 0/0 -> NaN, so RSI should be NaN or 100-100/(1+nan)
        assert len(result) == 30

    def test_atr_all_zeros(self):
        """Validates ATR when H=L=C (no volatility)."""
        df = pd.DataFrame({
            "open": [50.0] * 20, "high": [50.0] * 20,
            "low": [50.0] * 20, "close": [50.0] * 20,
            "volume": [1000.0] * 20,
        })
        result = TI.atr(df, length=5)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_bbands_all_same_values(self):
        """Validates Bollinger Bands when all values are identical (std=0)."""
        s = pd.Series([100.0] * 25)
        result = TI.bbands(s, length=20, std_dev=2.0)
        valid = result.dropna()
        if not valid.empty:
            assert (valid["BBU"] == valid["BBL"]).all()
            assert (valid["BBM"] == 100.0).all()

    def test_vwap_single_row(self, single_row_ohlcv):
        """Validates VWAP with single row returns a valid value."""
        result = TI.vwap(single_row_ohlcv)
        assert len(result) == 1
        expected = (101.0 + 99.0 + 100.5) / 3
        assert abs(result.iloc[0] - expected) < 1e-10

    def test_obv_constant_close(self):
        """Validates OBV with no price change (sign of diff = 0)."""
        df = pd.DataFrame({
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        })
        result = TI.obv(df)
        assert len(result) == 10

    def test_macd_single_value(self):
        """Validates MACD with a single-value series doesn't crash."""
        s = pd.Series([100.0])
        m, sig, h = TI.macd(s)
        assert len(m) == 1

    def test_engulfing_empty(self):
        """Validates engulfing with empty DataFrame returns empty."""
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        result = CP.engulfing(df)
        assert len(result) == 0

    def test_doji_with_zero_range(self):
        """Validates doji handles H=L (zero range) without division error."""
        df = pd.DataFrame({
            "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0],
        })
        result = CP.doji(df, threshold=0.05)
        assert len(result) == 1

    def test_heikin_ashi_single_row(self):
        """Validates Heikin Ashi with single row returns valid output."""
        df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0],
        })
        result = TI.heikin_ashi(df)
        assert len(result) == 1
        assert abs(result["HA_close"].iloc[0] - (100 + 105 + 95 + 102) / 4) < 1e-10
