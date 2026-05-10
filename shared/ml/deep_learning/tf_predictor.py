"""
TensorFlow/Keras LSTM Predictor
==================================

Usage:
    from shared.ml.deep_learning.tf_predictor import TFPredictor
    predictor = TFPredictor()
    predictor.train(df)
    signals = predictor.predict(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    logger.warning("TensorFlow not installed. Install: pip install tensorflow")


@dataclass
class TFConfig:
    """Configuration for TF LSTM predictor."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 60
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001


class TFPredictor:
    """TensorFlow/Keras LSTM predictor. Same interface as LSTMPredictor."""

    def __init__(self, config: Optional[TFConfig] = None):
        if not _HAS_TF:
            raise ImportError("TensorFlow required. Install: pip install tensorflow")
        self.config = config or TFConfig()
        self.model = None
        self.feature_cols = None
        self._scaler_mean = None
        self._scaler_std = None

    def _build_model(self, input_shape):
        """Build Keras Sequential LSTM model."""
        model = tf.keras.Sequential()
        for i in range(self.config.num_layers):
            return_seq = i < self.config.num_layers - 1
            if i == 0:
                model.add(tf.keras.layers.LSTM(
                    self.config.hidden_size,
                    return_sequences=return_seq,
                    input_shape=input_shape,
                ))
            else:
                model.add(tf.keras.layers.LSTM(
                    self.config.hidden_size,
                    return_sequences=return_seq,
                ))
            model.add(tf.keras.layers.Dropout(self.config.dropout))
        model.add(tf.keras.layers.Dense(1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
        )
        return model

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the TF LSTM model on OHLCV data."""
        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.compute_features(df)
        close = df["close"].reindex(features.index)
        target = close.pct_change().shift(-1).reindex(features.index).dropna()
        features = features.loc[target.index]

        self.feature_cols = list(features.columns)
        self._scaler_mean = features.mean()
        self._scaler_std = features.std().replace(0, 1)
        features_norm = (features - self._scaler_mean) / self._scaler_std

        X, y = fe.prepare_sequences(features_norm, target, self.config.seq_len)
        splits = fe.temporal_split(X, y)

        self.model = self._build_model((X.shape[1], X.shape[2]))

        history = self.model.fit(
            splits["X_train"], splits["y_train"],
            validation_data=(splits["X_val"], splits["y_val"]),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0,
        )

        return {
            "train_loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate prediction for the latest data point."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.compute_features(df)[self.feature_cols]
        features_norm = (features - self._scaler_mean) / self._scaler_std
        X_vals = features_norm.values.astype(np.float32)
        seq = X_vals[-self.config.seq_len:]
        pred = self.model.predict(seq.reshape(1, *seq.shape), verbose=0)
        return pd.Series(pred.flatten(), index=features.index[-1:], name="prediction")

    def save_model(self, path: str) -> None:
        """Save model."""
        import os
        if self.model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        import json
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "scaler_mean": self._scaler_mean.to_dict(),
                "scaler_std": self._scaler_std.to_dict(),
            }, f)

    def load_model(self, path: str) -> None:
        """Load model."""
        import json
        self.model = tf.keras.models.load_model(path)
        meta_path = path + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self._scaler_mean = pd.Series(meta["scaler_mean"])
        self._scaler_std = pd.Series(meta["scaler_std"])
