"""
Unified Technical Indicators Library
======================================

Three-tier indicator computation: TA-Lib (C, fastest) -> pandas-ta (Python)
-> manual NumPy/pandas fallback. All methods accept pd.DataFrame or pd.Series
and return pd.Series or pd.DataFrame.

Used by the ML regime classifier, backtester, and all Python strategies.

Usage:
    from shared.indicators.technical_indicators import TechnicalIndicators as TI
    rsi = TI.rsi(df["close"], 14)
    macd_line, signal, hist = TI.macd(df["close"])
    ichimoku = TI.ichimoku(df)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib  # type: ignore[import-untyped]

    _HAS_TALIB = True
    logger.debug("TA-Lib (C) available - using C-accelerated indicators")
except ImportError:
    _HAS_TALIB = False
    logger.debug("TA-Lib not installed - falling back to pandas-ta / manual")

try:
    import pandas_ta as ta  # type: ignore[import-untyped]

    _HAS_PANDAS_TA = True
    logger.debug("pandas-ta available - using accelerated indicators")
except ImportError:
    _HAS_PANDAS_TA = False
    logger.debug("pandas-ta not installed - using manual fallback calculations")


class TechnicalIndicators:
    """Static-method library of technical indicators.

    Three-tier priority: TA-Lib (C) -> pandas-ta -> manual NumPy/pandas.
    TA-Lib provides 10-100x speedup for compute-heavy indicators.
    """

    # ================================================================
    # TREND INDICATORS
    # ================================================================

    @staticmethod
    def sma(series: pd.Series, length: int = 20) -> pd.Series:
        """Simple Moving Average."""
        if _HAS_TALIB:
            result = talib.SMA(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="SMA")
        if _HAS_PANDAS_TA:
            result = ta.sma(series, length=length)
            return result if result is not None else series.rolling(window=length).mean()
        return series.rolling(window=length).mean()

    @staticmethod
    def ema(series: pd.Series, length: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        if _HAS_TALIB:
            result = talib.EMA(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="EMA")
        if _HAS_PANDAS_TA:
            result = ta.ema(series, length=length)
            return result if result is not None else series.ewm(span=length, adjust=False).mean()
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def dema(series: pd.Series, length: int = 20) -> pd.Series:
        """Double Exponential Moving Average."""
        if _HAS_TALIB:
            result = talib.DEMA(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="DEMA")
        if _HAS_PANDAS_TA:
            result = ta.dema(series, length=length)
            if result is not None:
                return result
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema2

    @staticmethod
    def tema(series: pd.Series, length: int = 20) -> pd.Series:
        """Triple Exponential Moving Average."""
        if _HAS_TALIB:
            result = talib.TEMA(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="TEMA")
        if _HAS_PANDAS_TA:
            result = ta.tema(series, length=length)
            if result is not None:
                return result
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    def wma(series: pd.Series, length: int = 20) -> pd.Series:
        """Weighted Moving Average."""
        if _HAS_TALIB:
            result = talib.WMA(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="WMA")
        if _HAS_PANDAS_TA:
            result = ta.wma(series, length=length)
            if result is not None:
                return result
        weights = np.arange(1, length + 1, dtype=float)
        return series.rolling(window=length).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    @staticmethod
    def kama(series: pd.Series, length: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average."""
        if _HAS_TALIB:
            result = talib.KAMA(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="KAMA")
        if _HAS_PANDAS_TA:
            result = ta.kama(series, length=length, fast=fast, slow=slow)
            if result is not None:
                return result
        values = series.values.astype(float)
        n = len(values)
        kama_arr = np.full(n, np.nan)
        if n <= length:
            return pd.Series(kama_arr, index=series.index)
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        kama_arr[length - 1] = values[length - 1]
        for i in range(length, n):
            direction = abs(values[i] - values[i - length])
            volatility = sum(abs(values[j] - values[j - 1]) for j in range(i - length + 1, i + 1))
            er = direction / volatility if volatility != 0 else 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama_arr[i] = kama_arr[i - 1] + sc * (values[i] - kama_arr[i - 1])
        return pd.Series(kama_arr, index=series.index, name="KAMA")

    @staticmethod
    def hma(series: pd.Series, length: int = 9) -> pd.Series:
        """Hull Moving Average."""
        if _HAS_PANDAS_TA:
            result = ta.hma(series, length=length)
            if result is not None:
                return result
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma_half = TechnicalIndicators.wma(series, half_len)
        wma_full = TechnicalIndicators.wma(series, length)
        diff = 2 * wma_half - wma_full
        return TechnicalIndicators.wma(diff, sqrt_len)

    @staticmethod
    def supertrend(
        df: pd.DataFrame, length: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        """Supertrend indicator. Returns DataFrame with SUPERT, SUPERTd, SUPERTl, SUPERTs."""
        if _HAS_PANDAS_TA:
            result = df.ta.supertrend(length=length, multiplier=multiplier)
            if result is not None:
                return result
        high, low, close = df["high"], df["low"], df["close"]
        atr = TechnicalIndicators.atr(df, length)
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        n = len(close)
        direction = np.ones(n)
        supertrend = np.full(n, np.nan)
        final_upper = upper.values.copy()
        final_lower = lower.values.copy()
        for i in range(1, n):
            if final_lower[i] < final_lower[i - 1] and close.iloc[i - 1] > final_lower[i - 1]:
                final_lower[i] = final_lower[i - 1]
            if final_upper[i] > final_upper[i - 1] and close.iloc[i - 1] < final_upper[i - 1]:
                final_upper[i] = final_upper[i - 1]
            if direction[i - 1] == 1:
                direction[i] = -1 if close.iloc[i] < final_lower[i] else 1
            else:
                direction[i] = 1 if close.iloc[i] > final_upper[i] else -1
            supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
        result_df = pd.DataFrame(index=df.index)
        result_df["SUPERT"] = supertrend
        result_df["SUPERTd"] = direction.astype(int)
        result_df["SUPERTl"] = np.where(direction == 1, final_lower, np.nan)
        result_df["SUPERTs"] = np.where(direction == -1, final_upper, np.nan)
        return result_df

    @staticmethod
    def ichimoku(
        df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52,
    ) -> pd.DataFrame:
        """Ichimoku Cloud. Returns DataFrame with tenkan_sen, kijun_sen,
        senkou_span_a, senkou_span_b, chikou_span."""
        if _HAS_PANDAS_TA:
            result = df.ta.ichimoku(tenkan=tenkan, kijun=kijun, senkou=senkou)
            if result is not None and isinstance(result, tuple):
                return result[0]
        high, low, close = df["high"], df["low"], df["close"]
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
        chikou = close.shift(-kijun)
        result_df = pd.DataFrame(index=df.index)
        result_df["tenkan_sen"] = tenkan_sen
        result_df["kijun_sen"] = kijun_sen
        result_df["senkou_span_a"] = senkou_a
        result_df["senkou_span_b"] = senkou_b
        result_df["chikou_span"] = chikou
        return result_df

    @staticmethod
    def psar(
        df: pd.DataFrame, af0: float = 0.02, af_step: float = 0.02, max_af: float = 0.20
    ) -> pd.Series:
        """Parabolic SAR."""
        if _HAS_TALIB:
            result = talib.SAR(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                acceleration=af0, maximum=max_af,
            )
            return pd.Series(result, index=df.index, name="PSAR")
        if _HAS_PANDAS_TA:
            result = ta.psar(
                high=df["high"], low=df["low"], close=df["close"],
                af0=af0, af=af_step, max_af=max_af,
            )
            if result is not None:
                long_col = [c for c in result.columns if "long" in c.lower()]
                short_col = [c for c in result.columns if "short" in c.lower()]
                if long_col and short_col:
                    return result[long_col[0]].fillna(result[short_col[0]])
                return result.iloc[:, 0]
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        n = len(high)
        psar_arr = np.full(n, np.nan)
        if n < 2:
            return pd.Series(psar_arr, index=df.index, name="PSAR")
        bull = True
        af = af0
        ep = high[0]
        psar_arr[0] = low[0]
        for i in range(1, n):
            if bull:
                psar_arr[i] = psar_arr[i - 1] + af * (ep - psar_arr[i - 1])
                psar_arr[i] = min(psar_arr[i], low[i - 1])
                if i >= 2:
                    psar_arr[i] = min(psar_arr[i], low[i - 2])
                if low[i] < psar_arr[i]:
                    bull = False
                    psar_arr[i] = ep
                    ep = low[i]
                    af = af0
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_step, max_af)
            else:
                psar_arr[i] = psar_arr[i - 1] + af * (ep - psar_arr[i - 1])
                psar_arr[i] = max(psar_arr[i], high[i - 1])
                if i >= 2:
                    psar_arr[i] = max(psar_arr[i], high[i - 2])
                if high[i] > psar_arr[i]:
                    bull = True
                    psar_arr[i] = ep
                    ep = high[i]
                    af = af0
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_step, max_af)
        return pd.Series(psar_arr, index=df.index, name="PSAR")

    # ================================================================
    # MOMENTUM INDICATORS
    # ================================================================

    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Relative Strength Index."""
        if _HAS_TALIB:
            result = talib.RSI(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="RSI")
        if _HAS_PANDAS_TA:
            result = ta.rsi(series, length=length)
            if result is not None:
                return result
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).rename("RSI")

    @staticmethod
    def stochastic(
        df: pd.DataFrame, k_length: int = 14, d_length: int = 3, smooth_k: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator. Returns (K, D)."""
        if _HAS_TALIB:
            k, d = talib.STOCH(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                fastk_period=k_length, slowk_period=smooth_k, slowd_period=d_length,
            )
            return (
                pd.Series(k, index=df.index, name="STOCH_K"),
                pd.Series(d, index=df.index, name="STOCH_D"),
            )
        if _HAS_PANDAS_TA:
            result = ta.stoch(
                high=df["high"], low=df["low"], close=df["close"],
                k=k_length, d=d_length, smooth_k=smooth_k,
            )
            if result is not None:
                return result.iloc[:, 0], result.iloc[:, 1]
        low_min = df["low"].rolling(window=k_length).min()
        high_max = df["high"].rolling(window=k_length).max()
        raw_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        k = raw_k.rolling(window=smooth_k).mean()
        d = k.rolling(window=d_length).mean()
        return k.rename("STOCH_K"), d.rename("STOCH_D")

    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD. Returns (macd_line, signal_line, histogram)."""
        if _HAS_TALIB:
            m, s, h = talib.MACD(
                series.values.astype(float),
                fastperiod=fast, slowperiod=slow, signalperiod=signal,
            )
            return (
                pd.Series(m, index=series.index, name="MACD"),
                pd.Series(s, index=series.index, name="MACDs"),
                pd.Series(h, index=series.index, name="MACDh"),
            )
        if _HAS_PANDAS_TA:
            result = ta.macd(series, fast=fast, slow=slow, signal=signal)
            if result is not None:
                return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line.rename("MACD"), signal_line.rename("MACDs"), histogram.rename("MACDh")

    @staticmethod
    def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        if _HAS_TALIB:
            result = talib.CCI(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=length,
            )
            return pd.Series(result, index=df.index, name="CCI")
        if _HAS_PANDAS_TA:
            result = ta.cci(high=df["high"], low=df["low"], close=df["close"], length=length)
            if result is not None:
                return result
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(window=length).mean()
        mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return ((tp - sma_tp) / (0.015 * mad)).rename("CCI")

    @staticmethod
    def williams_r(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Williams %R."""
        if _HAS_TALIB:
            result = talib.WILLR(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=length,
            )
            return pd.Series(result, index=df.index, name="WILLR")
        if _HAS_PANDAS_TA:
            result = ta.willr(high=df["high"], low=df["low"], close=df["close"], length=length)
            if result is not None:
                return result
        high_max = df["high"].rolling(window=length).max()
        low_min = df["low"].rolling(window=length).min()
        return (-100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)).rename("WILLR")

    @staticmethod
    def roc(series: pd.Series, length: int = 10) -> pd.Series:
        """Rate of Change."""
        if _HAS_TALIB:
            result = talib.ROC(series.values.astype(float), timeperiod=length)
            return pd.Series(result, index=series.index, name="ROC")
        if _HAS_PANDAS_TA:
            result = ta.roc(series, length=length)
            if result is not None:
                return result
        return ((series - series.shift(length)) / series.shift(length).replace(0, np.nan) * 100).rename("ROC")

    @staticmethod
    def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Money Flow Index."""
        if _HAS_TALIB:
            result = talib.MFI(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                df["volume"].values.astype(float),
                timeperiod=length,
            )
            return pd.Series(result, index=df.index, name="MFI")
        if _HAS_PANDAS_TA:
            result = ta.mfi(
                high=df["high"], low=df["low"], close=df["close"],
                volume=df["volume"], length=length,
            )
            if result is not None:
                return result
        tp = (df["high"] + df["low"] + df["close"]) / 3
        rmf = tp * df["volume"]
        delta = tp.diff()
        pos_flow = rmf.where(delta > 0, 0.0)
        neg_flow = rmf.where(delta < 0, 0.0)
        pos_sum = pos_flow.rolling(window=length).sum()
        neg_sum = neg_flow.rolling(window=length).sum()
        mr = pos_sum / neg_sum.replace(0, np.nan)
        return (100 - (100 / (1 + mr))).rename("MFI")

    @staticmethod
    def adx(df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index. Returns (ADX, +DI, -DI)."""
        if _HAS_TALIB:
            adx_val = talib.ADX(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=length,
            )
            plus_di = talib.PLUS_DI(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=length,
            )
            minus_di = talib.MINUS_DI(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=length,
            )
            return (
                pd.Series(adx_val, index=df.index, name="ADX"),
                pd.Series(plus_di, index=df.index, name="DMP"),
                pd.Series(minus_di, index=df.index, name="DMN"),
            )
        if _HAS_PANDAS_TA:
            result = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=length)
            if result is not None:
                return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = tr.ewm(alpha=1.0 / length, min_periods=length).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1.0 / length, min_periods=length).mean() / atr_val)
        minus_di = 100 * (minus_dm.ewm(alpha=1.0 / length, min_periods=length).mean() / atr_val)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.ewm(alpha=1.0 / length, min_periods=length).mean()
        return adx_val.rename("ADX"), plus_di.rename("DMP"), minus_di.rename("DMN")

    # ================================================================
    # VOLATILITY INDICATORS
    # ================================================================

    @staticmethod
    def bbands(
        series: pd.Series, length: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Bands. Returns DataFrame with BBU, BBM, BBL, BBB, BBP."""
        if _HAS_TALIB:
            upper, mid, lower = talib.BBANDS(
                series.values.astype(float),
                timeperiod=length, nbdevup=std_dev, nbdevdn=std_dev,
            )
            result_df = pd.DataFrame(index=series.index)
            result_df["BBL"] = lower
            result_df["BBM"] = mid
            result_df["BBU"] = upper
            result_df["BBB"] = (upper - lower) / np.where(mid != 0, mid, np.nan)
            result_df["BBP"] = (series.values - lower) / np.where(upper - lower != 0, upper - lower, np.nan)
            return result_df
        if _HAS_PANDAS_TA:
            result = ta.bbands(series, length=length, std=std_dev)
            if result is not None:
                return result
        mid = series.rolling(window=length).mean()
        std = series.rolling(window=length).std()
        upper = mid + std * std_dev
        lower = mid - std * std_dev
        width = (upper - lower) / mid
        pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
        result_df = pd.DataFrame(index=series.index)
        result_df["BBL"] = lower
        result_df["BBM"] = mid
        result_df["BBU"] = upper
        result_df["BBB"] = width
        result_df["BBP"] = pct_b
        return result_df

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Average True Range."""
        if _HAS_TALIB:
            result = talib.ATR(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=length,
            )
            return pd.Series(result, index=df.index, name="ATR")
        if _HAS_PANDAS_TA:
            result = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=length)
            if result is not None:
                return result
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / length, min_periods=length).mean().rename("ATR")

    @staticmethod
    def keltner_channels(
        df: pd.DataFrame, length: int = 20, multiplier: float = 1.5, atr_length: int = 10
    ) -> pd.DataFrame:
        """Keltner Channels. Returns DataFrame with KCU, KCM, KCL."""
        if _HAS_PANDAS_TA:
            result = df.ta.kc(length=length, scalar=multiplier, mamode="ema")
            if result is not None:
                return result
        mid = df["close"].ewm(span=length, adjust=False).mean()
        atr_val = TechnicalIndicators.atr(df, atr_length)
        upper = mid + multiplier * atr_val
        lower = mid - multiplier * atr_val
        result_df = pd.DataFrame(index=df.index)
        result_df["KCL"] = lower
        result_df["KCM"] = mid
        result_df["KCU"] = upper
        return result_df

    @staticmethod
    def donchian_channels(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
        """Donchian Channels. Returns DataFrame with DCU, DCM, DCL."""
        if _HAS_PANDAS_TA:
            result = df.ta.donchian(lower_length=length, upper_length=length)
            if result is not None:
                return result
        upper = df["high"].rolling(window=length).max()
        lower = df["low"].rolling(window=length).min()
        mid = (upper + lower) / 2
        result_df = pd.DataFrame(index=df.index)
        result_df["DCL"] = lower
        result_df["DCM"] = mid
        result_df["DCU"] = upper
        return result_df

    @staticmethod
    def chaikin_volatility(df: pd.DataFrame, length: int = 10, roc_length: int = 10) -> pd.Series:
        """Chaikin Volatility."""
        hl_diff = df["high"] - df["low"]
        ema_hl = hl_diff.ewm(span=length, adjust=False).mean()
        chaikin = ((ema_hl - ema_hl.shift(roc_length)) / ema_hl.shift(roc_length).replace(0, np.nan)) * 100
        return chaikin.rename("CHAIKIN_VOL")

    # ================================================================
    # VOLUME INDICATORS
    # ================================================================

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume."""
        if _HAS_TALIB:
            result = talib.OBV(
                df["close"].values.astype(float),
                df["volume"].values.astype(float),
            )
            return pd.Series(result, index=df.index, name="OBV")
        if _HAS_PANDAS_TA:
            result = ta.obv(close=df["close"], volume=df["volume"])
            if result is not None:
                return result
        sign = np.sign(df["close"].diff())
        return (sign * df["volume"]).cumsum().rename("OBV")

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price (cumulative intraday)."""
        if _HAS_PANDAS_TA:
            result = ta.vwap(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
            )
            if result is not None:
                return result
        tp = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (tp * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        return (cum_tp_vol / cum_vol.replace(0, np.nan)).rename("VWAP")

    @staticmethod
    def ad_line(df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line."""
        if _HAS_TALIB:
            result = talib.AD(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                df["volume"].values.astype(float),
            )
            return pd.Series(result, index=df.index, name="AD")
        if _HAS_PANDAS_TA:
            result = ta.ad(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"])
            if result is not None:
                return result
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
        return (clv * df["volume"]).cumsum().rename("AD")

    @staticmethod
    def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        if _HAS_PANDAS_TA:
            result = ta.cmf(
                high=df["high"], low=df["low"], close=df["close"],
                volume=df["volume"], length=length,
            )
            if result is not None:
                return result
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
        mfv = clv * df["volume"]
        return (mfv.rolling(window=length).sum() / df["volume"].rolling(window=length).sum().replace(0, np.nan)).rename("CMF")

    @staticmethod
    def volume_profile(df: pd.DataFrame, bins: int = 30) -> dict:
        """Volume Profile. Returns dict with poc, vah, val, and profile DataFrame."""
        price_min = df["low"].min()
        price_max = df["high"].max()
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_at_price = np.zeros(bins)
        for i in range(bins):
            mask = (df["close"] >= bin_edges[i]) & (df["close"] < bin_edges[i + 1])
            vol_at_price[i] = df.loc[mask, "volume"].sum()
        poc_idx = int(np.argmax(vol_at_price))
        poc = float(bin_centers[poc_idx])
        total_vol = vol_at_price.sum()
        target_vol = total_vol * 0.70
        sorted_idx = np.argsort(-vol_at_price)
        cum_vol = 0.0
        value_indices = []
        for idx in sorted_idx:
            cum_vol += vol_at_price[idx]
            value_indices.append(idx)
            if cum_vol >= target_vol:
                break
        val_idx = min(value_indices)
        vah_idx = max(value_indices)
        profile_df = pd.DataFrame({"price": bin_centers, "volume": vol_at_price})
        return {
            "poc": poc,
            "vah": float(bin_centers[vah_idx]),
            "val": float(bin_centers[val_idx]),
            "profile": profile_df,
        }

    # ================================================================
    # CUSTOM INDICATORS
    # ================================================================

    @staticmethod
    def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """Heikin Ashi candles. Returns DataFrame with HA_open, HA_high, HA_low, HA_close."""
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open = pd.Series(np.nan, index=df.index, dtype=float)
        ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
        ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
        result = pd.DataFrame(index=df.index)
        result["HA_open"] = ha_open
        result["HA_high"] = ha_high
        result["HA_low"] = ha_low
        result["HA_close"] = ha_close
        return result

    @staticmethod
    def pivot_points(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """Pivot Points (standard, fibonacci, camarilla).

        Uses the previous bar's high, low, close to compute pivot levels.
        Returns DataFrame with PP, S1-S3, R1-R3.
        """
        high = df["high"].shift(1)
        low = df["low"].shift(1)
        close = df["close"].shift(1)
        pp = (high + low + close) / 3
        result = pd.DataFrame(index=df.index)
        result["PP"] = pp
        if method == "standard":
            result["R1"] = 2 * pp - low
            result["S1"] = 2 * pp - high
            result["R2"] = pp + (high - low)
            result["S2"] = pp - (high - low)
            result["R3"] = high + 2 * (pp - low)
            result["S3"] = low - 2 * (high - pp)
        elif method == "fibonacci":
            diff = high - low
            result["R1"] = pp + 0.382 * diff
            result["S1"] = pp - 0.382 * diff
            result["R2"] = pp + 0.618 * diff
            result["S2"] = pp - 0.618 * diff
            result["R3"] = pp + diff
            result["S3"] = pp - diff
        elif method == "camarilla":
            diff = high - low
            result["R1"] = close + diff * 1.1 / 12
            result["S1"] = close - diff * 1.1 / 12
            result["R2"] = close + diff * 1.1 / 6
            result["S2"] = close - diff * 1.1 / 6
            result["R3"] = close + diff * 1.1 / 4
            result["S3"] = close - diff * 1.1 / 4
        return result
