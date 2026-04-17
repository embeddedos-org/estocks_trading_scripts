"""
Transformer Price Predictor (PyTorch)
=======================================

Usage:
    from shared.ml.deep_learning.transformer_predictor import TransformerPredictor
    predictor = TransformerPredictor()
    predictor.train(df)
    signals = predictor.predict(df)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class TransformerConfig:
    """Configuration for Transformer predictor."""
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    seq_len: int = 60
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "auto"


if _HAS_TORCH:

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class TimeSeriesTransformer(nn.Module):
        """Transformer model for time series prediction."""

        def __init__(self, input_size: int, config: TransformerConfig):
            super().__init__()
            self.input_proj = nn.Linear(input_size, config.d_model)
            self.pos_enc = PositionalEncoding(config.d_model, config.seq_len, config.dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.fc = nn.Linear(config.d_model, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            x = x[:, -1, :]
            return self.fc(x).squeeze(-1)


class TransformerPredictor:
    """Transformer predictor for stock prices. Same interface as LSTMPredictor."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required. Install: pip install torch")
        self.config = config or TransformerConfig()
        self.model = None
        self.feature_cols = None
        self._scaler_mean = None
        self._scaler_std = None
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the Transformer model on OHLCV data."""
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

        self.model = TimeSeriesTransformer(X.shape[2], self.config).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(torch.FloatTensor(splits["X_train"]), torch.FloatTensor(splits["y_train"]))
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        best_val_loss = float("inf")
        for epoch in range(self.config.epochs):
            self.model.train()
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(splits["X_val"]).to(self.device)
                val_y = torch.FloatTensor(splits["y_val"]).to(self.device)
                val_loss = criterion(self.model(val_x), val_y).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d - train: %.6f, val: %.6f", epoch + 1, self.config.epochs, np.mean(losses), val_loss)

        return {"train_loss": np.mean(losses), "val_loss": best_val_loss}

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
        x_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()
        return pd.Series(pred, index=features.index[-1:], name="prediction")

    def save_model(self, path: str) -> None:
        """Save model weights and config."""
        import os
        if self.model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "feature_cols": self.feature_cols,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }, path)

    def load_model(self, path: str) -> None:
        """Load model weights and config."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint["config"]
        self.feature_cols = checkpoint["feature_cols"]
        self._scaler_mean = checkpoint["scaler_mean"]
        self._scaler_std = checkpoint["scaler_std"]
        self.model = TimeSeriesTransformer(len(self.feature_cols), self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
