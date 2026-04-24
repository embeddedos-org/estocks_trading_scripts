"""
ML and RL Strategy Examples
==============================

Advanced strategies using machine learning for signal generation.

Demonstrates:
- LSTM-based price prediction (MLStrategy)
- RL-based trading agent (RLStrategy)
- Graceful fallback when torch/SB3 not installed

Usage:
    from strategies.examples.ml_rl_strategy import MLStrategy, RLStrategy
    ml = MLStrategy()
    ml.train(df)
    engine = BacktestEngineV2()
    engine.load_data(df)
    result = engine.run(ml.generate_signals)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.backtesting.backtest_engine_v2 import (
    BacktestContext,
    BacktestEngineV2,
    BacktestResultV2,
)
from shared.indicators.technical_indicators import TechnicalIndicators as TI
from strategies import register_strategy

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

_HAS_SB3 = False
try:
    from stable_baselines3 import PPO  # type: ignore[import-untyped]
    _HAS_SB3 = True
except ImportError:
    pass


@dataclass
class MLConfig:
    """Configuration for MLStrategy."""

    seq_len: int = 60
    epochs: int = 20
    hidden_size: int = 64
    threshold: float = 0.0
    prediction_threshold: float = 0.005
    trailing_stop_pct: float = 0.05
    use_enricher: bool = True


@register_strategy("ml")
class MLStrategy:
    """LSTM-based price prediction strategy.

    Trains an LSTM on historical features to predict next-day returns.
    Signals: predicted return > 0 -> buy, < 0 -> sell.
    Falls back to simple momentum if PyTorch is not installed.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()
        self._predictor = None
        self._feature_engineer = None
        self._fallback = not _HAS_TORCH
        self._entry_prices: Dict[str, float] = {}
        self._peak_prices: Dict[str, float] = {}
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception:
                pass

        if _HAS_TORCH:
            try:
                from shared.ml.deep_learning.feature_engineer import FeatureEngineer
                from shared.ml.deep_learning.lstm_predictor import LSTMPredictor, LSTMConfig
                self._feature_engineer = FeatureEngineer()
                lstm_config = LSTMConfig(
                    hidden_size=self.config.hidden_size,
                    seq_len=self.config.seq_len,
                    epochs=self.config.epochs,
                )
                self._predictor = LSTMPredictor(lstm_config)
            except Exception:
                self._fallback = True

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MLStrategy":
        config = MLConfig(**{
            k: v for k, v in params.items() if hasattr(MLConfig, k)
        })
        return cls(config)

    def train(self, df: pd.DataFrame) -> None:
        """Train the LSTM model on historical data."""
        if self._fallback:
            print("  [ML] PyTorch not available - using momentum fallback")
            return

        print("  [ML] Computing features...")
        features = self._feature_engineer.compute_features(df)
        target = df["close"].pct_change().shift(-1).loc[features.index]
        valid = features.index.intersection(target.dropna().index)
        features = features.loc[valid]
        target = target.loc[valid]

        print(f"  [ML] Training LSTM on {len(features)} samples...")
        self._predictor.train(features, target)
        print("  [ML] Training complete.")

    def generate_signals(self, ctx: BacktestContext) -> Dict[str, int]:
        """Generate signals using LSTM predictions or momentum fallback."""
        signals: Dict[str, int] = {}
        threshold = self.config.prediction_threshold

        for sym, df in ctx.bars.items():
            if len(df) < 60:
                signals[sym] = 0
                continue

            current_price = float(df["close"].iloc[-1])
            current_pos = ctx.positions.get(sym, 0)

            # Trailing stop check for held positions
            if current_pos != 0 and sym in self._entry_prices:
                entry = self._entry_prices[sym]
                # Update peak price for trailing stop
                if current_pos > 0:
                    self._peak_prices[sym] = max(self._peak_prices.get(sym, entry), current_price)
                    if current_price < self._peak_prices[sym] * (1 - self.config.trailing_stop_pct):
                        signals[sym] = 0
                        self._entry_prices.pop(sym, None)
                        self._peak_prices.pop(sym, None)
                        continue
                elif current_pos < 0:
                    self._peak_prices[sym] = min(self._peak_prices.get(sym, entry), current_price)
                    if current_price > self._peak_prices[sym] * (1 + self.config.trailing_stop_pct):
                        signals[sym] = 0
                        self._entry_prices.pop(sym, None)
                        self._peak_prices.pop(sym, None)
                        continue

            if self._fallback:
                # Enricher gate for new entries
                enricher_ok = True
                if getattr(self, "_enricher", None) and current_pos == 0:
                    enriched = self._enricher.enrich(sym, df)
                    blocked, _ = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False

                close = df["close"]
                mom_20 = float(close.iloc[-1] / close.iloc[-20] - 1) if len(df) >= 20 else 0
                mom_5 = float(close.iloc[-1] / close.iloc[-5] - 1) if len(df) >= 5 else 0
                score = 0.6 * mom_20 + 0.4 * mom_5
                if score > 0.01 and enricher_ok:
                    signals[sym] = 1
                    if current_pos <= 0:
                        self._entry_prices[sym] = current_price
                        self._peak_prices[sym] = current_price
                elif score < -0.01:
                    signals[sym] = -1
                    if current_pos >= 0:
                        self._entry_prices[sym] = current_price
                        self._peak_prices[sym] = current_price
                else:
                    signals[sym] = 0
            else:
                try:
                    # Enricher gate for new entries
                    enricher_ok = True
                    if getattr(self, "_enricher", None) and current_pos == 0:
                        enriched = self._enricher.enrich(sym, df)
                        blocked, _ = self._enricher.should_block_entry(enriched)
                        if blocked:
                            enricher_ok = False

                    preds = self._predictor.predict(df)
                    if preds is not None and len(preds) > 0:
                        pred = float(preds[-1]) if hasattr(preds, '__len__') else float(preds)
                        if pred > threshold and enricher_ok:
                            signals[sym] = 1
                            if current_pos <= 0:
                                self._entry_prices[sym] = current_price
                                self._peak_prices[sym] = current_price
                        elif pred < -threshold:
                            signals[sym] = -1
                            if current_pos >= 0:
                                self._entry_prices[sym] = current_price
                                self._peak_prices[sym] = current_price
                        else:
                            signals[sym] = 0
                    else:
                        signals[sym] = 0
                except Exception:
                    signals[sym] = 0

        return signals


@dataclass
class RLConfig:
    """Configuration for RLStrategy."""

    algorithm: str = "PPO"
    total_timesteps: int = 50_000
    reward_type: str = "pnl"
    trailing_stop_pct: float = 0.05  # 5% trailing stop
    use_enricher: bool = True


@register_strategy("rl")
class RLStrategy:
    """RL-based trading agent strategy.

    Trains a PPO agent via TradingEnv gymnasium environment.
    Falls back to RSI-based signals if stable-baselines3 is not installed.
    """

    def __init__(self, config: RLConfig | None = None) -> None:
        self.config = config or RLConfig()
        self._trader = None
        self._fallback = not _HAS_SB3
        self._entry_prices: Dict[str, float] = {}
        self._peak_prices: Dict[str, float] = {}
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception:
                pass

        if _HAS_SB3:
            try:
                from shared.ml.rl_agent import RLTrader
                self._trader = RLTrader(algorithm=self.config.algorithm)
            except Exception:
                self._fallback = True

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "RLStrategy":
        config = RLConfig(**{
            k: v for k, v in params.items() if hasattr(RLConfig, k)
        })
        return cls(config)

    def train(self, df: pd.DataFrame) -> None:
        """Train the RL agent on historical data."""
        if self._fallback:
            print("  [RL] stable-baselines3 not available - using RSI fallback")
            return

        print(f"  [RL] Training {self.config.algorithm} for {self.config.total_timesteps} timesteps...")
        self._trader.train(df, total_timesteps=self.config.total_timesteps)
        print("  [RL] Training complete.")

    def generate_signals(self, ctx: BacktestContext) -> Dict[str, int]:
        """Generate signals using RL agent or RSI fallback."""
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < 20:
                signals[sym] = 0
                continue

            current_price = float(df["close"].iloc[-1])
            current_pos = ctx.positions.get(sym, 0)

            # Trailing stop check for held positions
            if current_pos != 0 and sym in self._entry_prices:
                entry = self._entry_prices[sym]
                if current_pos > 0:
                    self._peak_prices[sym] = max(self._peak_prices.get(sym, entry), current_price)
                    if current_price < self._peak_prices[sym] * (1 - self.config.trailing_stop_pct):
                        signals[sym] = 0
                        self._entry_prices.pop(sym, None)
                        self._peak_prices.pop(sym, None)
                        continue
                elif current_pos < 0:
                    self._peak_prices[sym] = min(self._peak_prices.get(sym, entry), current_price)
                    if current_price > self._peak_prices[sym] * (1 + self.config.trailing_stop_pct):
                        signals[sym] = 0
                        self._entry_prices.pop(sym, None)
                        self._peak_prices.pop(sym, None)
                        continue

            if self._fallback:
                # Enricher gate for new entries
                enricher_ok = True
                if getattr(self, "_enricher", None) and current_pos == 0:
                    enriched = self._enricher.enrich(sym, df)
                    blocked, _ = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False

                close = df["close"]
                rsi = TI.rsi(close, 14)
                current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
                if current_rsi < 35 and enricher_ok:
                    signals[sym] = 1
                    if current_pos <= 0:
                        self._entry_prices[sym] = current_price
                        self._peak_prices[sym] = current_price
                elif current_rsi > 65:
                    signals[sym] = -1
                    if current_pos >= 0:
                        self._entry_prices[sym] = current_price
                        self._peak_prices[sym] = current_price
                else:
                    pos = ctx.positions.get(sym, 0)
                    signals[sym] = 1 if pos > 0 else (-1 if pos < 0 else 0)
            else:
                try:
                    action = self._trader.predict(df)
                    sig = int(action)
                    signals[sym] = sig
                    if sig == 1 and current_pos <= 0:
                        self._entry_prices[sym] = current_price
                        self._peak_prices[sym] = current_price
                    elif sig == -1 and current_pos >= 0:
                        self._entry_prices[sym] = current_price
                        self._peak_prices[sym] = current_price
                except Exception:
                    signals[sym] = 0

        return signals


def _generate_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for ML/RL testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    price = 100.0
    prices = []
    for i in range(n_bars):
        regime = np.sin(2 * np.pi * i / 120)
        drift = 0.0003 * regime
        ret = drift + rng.randn() * 0.015
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.005)
        low = price * (1 - abs(rng.randn()) * 0.005)
        prices.append({
            "date": dates[i],
            "open": price * (1 + rng.randn() * 0.002),
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(prices)


def run_example() -> BacktestResultV2:
    """Run ML and RL strategies on synthetic data."""
    print("=" * 60)
    print("ML / RL STRATEGY EXAMPLE")
    print("=" * 60)

    df = _generate_data()

    # --- ML Strategy ---
    print("\n--- ML Strategy (LSTM) ---")
    ml = MLStrategy()
    ml.train(df)

    engine = BacktestEngineV2(initial_capital=100_000)
    engine.load_data(df)
    ml_result = engine.run(ml.generate_signals)

    print(f"  Total Return:  {ml_result.total_return:>10.2%}")
    print(f"  Sharpe Ratio:  {ml_result.sharpe_ratio:>10.4f}")
    print(f"  Max Drawdown:  {ml_result.max_drawdown:>10.2%}")
    print(f"  Total Trades:  {ml_result.total_trades:>10d}")
    print(f"  Win Rate:      {ml_result.win_rate:>10.2%}")

    # --- RL Strategy ---
    print("\n--- RL Strategy (PPO) ---")
    rl = RLStrategy()
    rl.train(df)

    engine2 = BacktestEngineV2(initial_capital=100_000)
    engine2.load_data(df)
    rl_result = engine2.run(rl.generate_signals)

    print(f"  Total Return:  {rl_result.total_return:>10.2%}")
    print(f"  Sharpe Ratio:  {rl_result.sharpe_ratio:>10.4f}")
    print(f"  Max Drawdown:  {rl_result.max_drawdown:>10.2%}")
    print(f"  Total Trades:  {rl_result.total_trades:>10d}")
    print(f"  Win Rate:      {rl_result.win_rate:>10.2%}")

    print("\n" + "=" * 60)
    return ml_result


if __name__ == "__main__":
    run_example()
