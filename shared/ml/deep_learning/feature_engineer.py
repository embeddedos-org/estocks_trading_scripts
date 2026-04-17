"""
Feature Engineering for Deep Learning
========================================

Computes features for LSTM/Transformer price prediction.

Usage:
    from shared.ml.deep_learning.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    features = fe.compute_features(df)
    X_seq, y_seq = fe.prepare_sequences(features, target, seq_len=60)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for deep learning price prediction."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from OHLCV data.

        Includes MLRegimeClassifier features plus lags, rolling skew/kurtosis.
        """
        features = pd.DataFrame(index=df.index)
        try:
            from shared.ml.regime_classifier import MLRegimeClassifier
            clf = MLRegimeClassifier()
            base = clf.compute_features(df)
            features = features.join(base)
        except ImportError:
            close = df["close"]
            features["ret_1d"] = close.pct_change(1)
            features["ret_5d"] = close.pct_change(5)
            features["ret_10d"] = close.pct_change(10)
            features["ret_20d"] = close.pct_change(20)
            features["vol_5d"] = features["ret_1d"].rolling(5).std()
            features["vol_10d"] = features["ret_1d"].rolling(10).std()
            features["vol_20d"] = features["ret_1d"].rolling(20).std()

        close = df["close"]
        for lag in [1, 2, 3, 5, 10, 21]:
            features[f"lag_{lag}"] = close.pct_change(lag)

        ret = close.pct_change()
        for window in [10, 20, 60]:
            features[f"skew_{window}"] = ret.rolling(window).skew()
            features[f"kurt_{window}"] = ret.rolling(window).kurt()

        for period in [5, 10, 20, 50]:
            sma = close.rolling(period).mean()
            features[f"price_sma_ratio_{period}"] = close / sma.replace(0, np.nan)

        if "volume" in df.columns:
            vol = df["volume"]
            features["vol_sma_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)
            features["vol_change"] = vol.pct_change()

        if all(c in df.columns for c in ["high", "low"]):
            features["hl_ratio"] = (df["high"] - df["low"]) / close.replace(0, np.nan)
            features["oc_ratio"] = (df["close"] - df["open"]).abs() / close.replace(0, np.nan)

        return features.dropna()

    def select_features(self, X: pd.DataFrame, y: pd.Series, top_n: int = 50) -> list:
        """Select top features by LightGBM importance."""
        try:
            import lightgbm as lgb
        except ImportError:
            return list(X.columns[:top_n])
        model = lgb.LGBMRegressor(n_estimators=100, verbose=-1)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        return importance.nlargest(min(top_n, len(importance))).index.tolist()

    @staticmethod
    def prepare_sequences(features: pd.DataFrame, target: pd.Series, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM/Transformer input."""
        X_vals = features.values.astype(np.float32)
        y_vals = target.values.astype(np.float32)
        n = len(X_vals) - seq_len
        if n <= 0:
            raise ValueError(f"Not enough data: {len(X_vals)} rows, need > {seq_len}")
        X_seq = np.zeros((n, seq_len, X_vals.shape[1]), dtype=np.float32)
        y_seq = np.zeros(n, dtype=np.float32)
        for i in range(n):
            X_seq[i] = X_vals[i : i + seq_len]
            y_seq[i] = y_vals[i + seq_len]
        return X_seq, y_seq

    @staticmethod
    def temporal_split(X: np.ndarray, y: np.ndarray, train_pct: float = 0.7, val_pct: float = 0.15) -> dict:
        """Chronological train/val/test split (no lookahead)."""
        n = len(X)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        return {
            "X_train": X[:train_end], "y_train": y[:train_end],
            "X_val": X[train_end:val_end], "y_val": y[train_end:val_end],
            "X_test": X[val_end:], "y_test": y[val_end:],
        }
