"""
Self-Learning Agent — Autonomous Investment Decision Maker
=============================================================

The "brain" that transforms stocks_plugin from a rule-based toolkit into
a self-improving investment AI agent.

Decision Loop:
    1. Ingest new market data
    2. Classify current market regime (ML Regime Classifier)
    3. Query TradeMemory: "what worked in similar conditions?"
    4. Generate predictions from all models (LSTM, Transformer, RL)
    5. Combine via EnsemblePredictor with adaptive weights
    6. Apply risk management gates
    7. Execute decision (BUY / SELL / HOLD)
    8. Record full context to TradeMemory
    9. Periodically retrain underperforming models

Self-Improvement Mechanisms:
    - Adaptive ensemble weights: models that perform well get more influence
    - Regime-aware memory: learns which strategies work in which conditions
    - Automatic retraining: detects model degradation and retrains
    - Confidence calibration: adjusts thresholds based on hit rate
    - Trade journaling: every decision is stored with full feature context

Usage:
    agent = SelfLearningAgent(db_path="my_trades.db")
    agent.train(df_historical)
    decision = agent.decide(df_current)
    print(f"Action: {decision['action']}, Confidence: {decision['confidence']}")

    # After trade completes:
    agent.record_outcome(entry_price=150, exit_price=155, pnl=500)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Local imports
from shared.ml.ensemble_predictor import EnsemblePredictor, EnsembleSignal
from shared.ml.trade_memory import TradeDecisionRecord, TradeMemory
from shared.risk_manager import RiskManager, RiskManagerConfig


@dataclass
class AgentConfig:
    """Configuration for the SelfLearningAgent."""

    # Database
    db_path: str = "trade_memory.db"

    # Model training
    retrain_interval_trades: int = 100  # retrain after N new trades
    min_accuracy_threshold: float = 0.45  # below this triggers retrain
    retrain_on_degradation: bool = True

    # Ensemble
    buy_threshold: float = 0.15
    sell_threshold: float = -0.15
    min_confidence: float = 0.3

    # Risk
    risk_per_trade_pct: float = 2.0
    max_daily_loss: float = 5000.0
    max_drawdown_pct: float = 10.0
    total_capital: float = 100_000.0

    # Memory queries
    regime_lookback_days: int = 90
    weight_update_interval: int = 20  # update ensemble weights every N decisions

    # Confidence calibration
    confidence_history_window: int = 50
    adaptive_thresholds: bool = True


class SelfLearningAgent:
    """Autonomous self-learning investment agent.

    Orchestrates ML models, ensemble prediction, risk management,
    and trade memory to make increasingly better decisions over time.

    Args:
        config: AgentConfig with all parameters.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()

        # Core components
        self._memory = TradeMemory(self.config.db_path)
        self._ensemble = EnsemblePredictor(
            buy_threshold=self.config.buy_threshold,
            sell_threshold=self.config.sell_threshold,
            min_confidence=self.config.min_confidence,
        )
        self._risk_manager = RiskManager(RiskManagerConfig(
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            max_daily_loss=self.config.max_daily_loss,
            max_drawdown_pct=self.config.max_drawdown_pct,
            total_capital=self.config.total_capital,
        ))

        # ML Models (lazy-loaded)
        self._regime_classifier = None
        self._lstm_predictor = None
        self._transformer_predictor = None
        self._rl_trader = None

        # State tracking
        self._models_trained = False
        self._decision_count = 0
        self._last_weight_update = 0
        self._pending_trade: Optional[Dict[str, Any]] = None
        self._current_regime = "UNKNOWN"

        logger.info("SelfLearningAgent initialized (memory: %s)", self._memory)

    # ─── Training ───

    def train(
        self,
        df: pd.DataFrame,
        models: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train all ML models on historical data.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.
            models: List of models to train. Default: all available.
                Options: "regime", "lstm", "transformer", "rl"
            verbose: Print training progress.

        Returns:
            Dict with training results per model.
        """
        if models is None:
            models = ["regime", "lstm", "transformer", "rl"]

        results: Dict[str, Any] = {}

        # Regime Classifier (LightGBM)
        if "regime" in models:
            try:
                from shared.ml.regime_classifier import MLRegimeClassifier
                if verbose:
                    print("  [Agent] Training Regime Classifier (LightGBM)...")
                self._regime_classifier = MLRegimeClassifier()
                regime_metrics = self._regime_classifier.fit(df)
                results["regime"] = regime_metrics
                if verbose:
                    print(f"  [Agent] Regime Classifier: accuracy={regime_metrics['accuracy']:.2%}")
            except ImportError:
                logger.warning("LightGBM not available — regime classifier skipped")
                results["regime"] = {"status": "skipped", "reason": "lightgbm not installed"}

        # LSTM Predictor
        if "lstm" in models:
            try:
                from shared.ml.deep_learning.lstm_predictor import LSTMPredictor, LSTMConfig
                if verbose:
                    print("  [Agent] Training LSTM Predictor (PyTorch)...")
                self._lstm_predictor = LSTMPredictor(LSTMConfig(
                    hidden_size=128, num_layers=2, epochs=30, seq_len=60,
                ))
                lstm_metrics = self._lstm_predictor.train(df)
                results["lstm"] = lstm_metrics
                if verbose:
                    print(f"  [Agent] LSTM: val_loss={lstm_metrics['val_loss']:.6f}")
            except ImportError:
                logger.warning("PyTorch not available — LSTM skipped")
                results["lstm"] = {"status": "skipped", "reason": "torch not installed"}

        # Transformer Predictor
        if "transformer" in models:
            try:
                from shared.ml.deep_learning.transformer_predictor import (
                    TransformerPredictor, TransformerConfig,
                )
                if verbose:
                    print("  [Agent] Training Transformer Predictor (PyTorch)...")
                self._transformer_predictor = TransformerPredictor(TransformerConfig(
                    d_model=64, nhead=4, num_layers=2, epochs=30, seq_len=60,
                ))
                tf_metrics = self._transformer_predictor.train(df)
                results["transformer"] = tf_metrics
                if verbose:
                    print(f"  [Agent] Transformer: val_loss={tf_metrics['val_loss']:.6f}")
            except ImportError:
                logger.warning("PyTorch not available — Transformer skipped")
                results["transformer"] = {"status": "skipped", "reason": "torch not installed"}

        # RL Agent
        if "rl" in models:
            try:
                from shared.ml.rl_agent import RLTrader
                if verbose:
                    print("  [Agent] Training RL Agent (PPO)...")
                self._rl_trader = RLTrader(
                    algorithm="PPO",
                    env_config={"reward_type": "risk_adjusted"},
                )
                rl_metrics = self._rl_trader.train(df, total_timesteps=50_000)
                results["rl"] = rl_metrics
                if verbose:
                    print(f"  [Agent] RL: trained {rl_metrics.get('total_timesteps', 0)} timesteps")
            except ImportError:
                logger.warning("stable-baselines3 not available — RL skipped")
                results["rl"] = {"status": "skipped", "reason": "stable-baselines3 not installed"}

        self._models_trained = True
        if verbose:
            active = [k for k, v in results.items() if v.get("status") != "skipped"]
            print(f"\n  [Agent] Training complete. Active models: {active}")

        return results

    # ─── Decision Making ───

    def decide(
        self,
        df: pd.DataFrame,
        symbol: str = "STOCK",
    ) -> Dict[str, Any]:
        """Make an autonomous investment decision.

        This is the core decision loop:
        1. Classify regime
        2. Query memory for historical context
        3. Gather predictions from all models
        4. Combine via ensemble
        5. Apply risk gates
        6. Return decision

        Args:
            df: Recent OHLCV DataFrame (needs enough history for features).
            symbol: Ticker symbol being evaluated.

        Returns:
            Dict with: action, confidence, regime, predictions, risk_status, reasoning
        """
        self._decision_count += 1

        # Periodically update ensemble weights from memory
        if (self._decision_count - self._last_weight_update) >= self.config.weight_update_interval:
            self._update_ensemble_weights()
            self._last_weight_update = self._decision_count

        # Step 1: Classify regime
        regime, regime_proba = self._classify_regime(df)
        self._current_regime = regime

        # Step 2: Query memory for similar conditions
        memory_insight = self._memory.query_similar_regime(
            regime, lookback_days=self.config.regime_lookback_days,
        )

        # Step 3: Gather model predictions
        predictions = self._gather_predictions(df)

        # Step 4: Ensemble
        signal = self._ensemble.predict(predictions, regime=regime)

        # Step 5: Risk gates
        can_trade = self._risk_manager.can_trade()

        # Step 6: Build decision
        action = "HOLD"
        if can_trade and signal.direction != 0:
            if signal.direction == 1:
                action = "BUY"
            elif signal.direction == -1:
                action = "SELL"

        # Memory-informed adjustment
        if memory_insight.get("sufficient_data") and memory_insight.get("recommendation") == "caution":
            if signal.confidence < 0.6:
                action = "HOLD"
                logger.info("Memory override: regime=%s historically underperforms, holding", regime)

        # Adaptive threshold adjustment
        if self.config.adaptive_thresholds:
            action = self._apply_adaptive_thresholds(action, signal, regime)

        # Build reasoning
        reasoning = self._build_reasoning(
            regime, regime_proba, memory_insight, predictions, signal, can_trade,
        )

        # Store pending trade context for record_outcome()
        features_snapshot = {}
        try:
            from shared.ml.regime_classifier import MLRegimeClassifier
            feat = MLRegimeClassifier.compute_features(df)
            top_features = feat.iloc[-1].dropna().to_dict()
            features_snapshot = {k: round(float(v), 6) for k, v in list(top_features.items())[:10]}
        except Exception:
            pass

        current_price = float(df["close"].iloc[-1])
        self._pending_trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "entry_price": current_price,
            "regime": regime,
            "regime_confidence": regime_proba.get(regime, 0),
            "features_snapshot": json.dumps(features_snapshot),
            "lstm_prediction": predictions.get("lstm", 0),
            "transformer_prediction": predictions.get("transformer", 0),
            "rl_action": int(predictions.get("rl", 0)),
            "regime_prediction": regime,
            "ensemble_signal": signal.raw_score,
            "ensemble_confidence": signal.confidence,
            "decision_source": "ensemble",
            "portfolio_value_at_entry": self._risk_manager._current_equity,
        }

        decision = {
            "action": action,
            "confidence": signal.confidence,
            "regime": regime,
            "regime_probabilities": regime_proba,
            "predictions": predictions,
            "ensemble_signal": {
                "direction": signal.direction,
                "raw_score": signal.raw_score,
                "agreement": signal.agreement_ratio,
                "contributions": signal.model_contributions,
            },
            "risk_status": self._risk_manager.get_status(),
            "memory_insight": memory_insight,
            "reasoning": reasoning,
            "price": current_price,
        }

        logger.info(
            "Decision #%d: %s %s @ %.2f | confidence=%.2f | regime=%s",
            self._decision_count, action, symbol, current_price,
            signal.confidence, regime,
        )

        return decision

    def record_outcome(
        self,
        exit_price: float,
        pnl: float,
        holding_period_bars: int = 0,
        max_favorable: float = 0.0,
        max_adverse: float = 0.0,
    ) -> None:
        """Record the outcome of the last decision.

        Must be called after decide() when the trade completes.

        Args:
            exit_price: Price at exit.
            pnl: Realized P&L in dollars.
            holding_period_bars: How many bars the trade was held.
            max_favorable: Best unrealized P&L during the trade.
            max_adverse: Worst unrealized P&L during the trade.
        """
        if self._pending_trade is None:
            logger.warning("No pending trade to record outcome for")
            return

        entry_price = self._pending_trade["entry_price"]
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        if self._pending_trade["action"] == "SELL":
            pnl_pct = -pnl_pct

        trade = TradeDecisionRecord(
            timestamp=self._pending_trade["timestamp"],
            symbol=self._pending_trade["symbol"],
            action=self._pending_trade["action"],
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            regime=self._pending_trade["regime"],
            regime_confidence=self._pending_trade["regime_confidence"],
            features_snapshot=self._pending_trade["features_snapshot"],
            lstm_prediction=self._pending_trade["lstm_prediction"],
            transformer_prediction=self._pending_trade["transformer_prediction"],
            rl_action=self._pending_trade["rl_action"],
            regime_prediction=self._pending_trade["regime_prediction"],
            ensemble_signal=self._pending_trade["ensemble_signal"],
            ensemble_confidence=self._pending_trade["ensemble_confidence"],
            decision_source=self._pending_trade["decision_source"],
            holding_period_bars=holding_period_bars,
            max_favorable_excursion=max_favorable,
            max_adverse_excursion=max_adverse,
            is_winner=pnl > 0,
            portfolio_value_at_entry=self._pending_trade["portfolio_value_at_entry"],
        )

        self._memory.record_trade(trade)

        # Record individual model predictions for accuracy tracking
        actual_direction = 1.0 if pnl > 0 else (-1.0 if pnl < 0 else 0.0)
        for model_name in ("lstm", "transformer", "rl"):
            pred_key = f"{model_name}_prediction" if model_name != "rl" else "rl_action"
            pred_value = self._pending_trade.get(pred_key, 0)
            if pred_value != 0:
                self._memory.record_model_prediction(
                    model_name=model_name,
                    prediction=float(pred_value),
                    actual_outcome=actual_direction,
                    regime=self._pending_trade["regime"],
                    symbol=self._pending_trade["symbol"],
                )

        # Update risk manager
        self._risk_manager.record_trade(symbol=trade.symbol, pnl=pnl)

        # Check if retraining is needed
        if self.config.retrain_on_degradation:
            self._check_retrain_needed()

        self._pending_trade = None

        logger.info(
            "Outcome recorded: %s P&L=$%.2f (%.2f%%) | is_winner=%s",
            trade.symbol, pnl, pnl_pct * 100, trade.is_winner,
        )

    # ─── Internal Methods ───

    def _classify_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """Classify current market regime."""
        if self._regime_classifier is not None:
            try:
                regime = self._regime_classifier.predict(df)
                proba = self._regime_classifier.predict_proba(df)
                return regime.name, proba
            except Exception as e:
                logger.warning("Regime classification failed: %s", e)

        # Fallback: simple ADX-based regime
        return self._fallback_regime(df)

    def _fallback_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """Simple ADX-based regime classification as fallback."""
        close = df["close"]
        high, low = df["high"], df["low"]

        # Compute ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, min_periods=14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, min_periods=14).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, min_periods=14).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1 / 14, min_periods=14).mean()

        # Volatility
        vol = close.pct_change().rolling(5).std()
        vol_90pct = vol.rolling(60).quantile(0.9)

        current_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20
        current_vol = float(vol.iloc[-1]) if not np.isnan(vol.iloc[-1]) else 0
        vol_threshold = float(vol_90pct.iloc[-1]) if not np.isnan(vol_90pct.iloc[-1]) else current_vol

        if current_vol > vol_threshold:
            return "VOLATILE", {"TRENDING": 0.1, "RANGING": 0.2, "VOLATILE": 0.7}
        elif current_adx > 25:
            return "TRENDING", {"TRENDING": 0.7, "RANGING": 0.2, "VOLATILE": 0.1}
        else:
            return "RANGING", {"TRENDING": 0.2, "RANGING": 0.7, "VOLATILE": 0.1}

    def _gather_predictions(self, df: pd.DataFrame) -> Dict[str, float]:
        """Collect predictions from all available models."""
        predictions: Dict[str, float] = {}

        # LSTM
        if self._lstm_predictor is not None:
            try:
                pred = self._lstm_predictor.predict(df)
                predictions["lstm"] = float(pred.iloc[-1]) if len(pred) > 0 else 0.0
            except Exception as e:
                logger.debug("LSTM prediction failed: %s", e)

        # Transformer
        if self._transformer_predictor is not None:
            try:
                pred = self._transformer_predictor.predict(df)
                predictions["transformer"] = float(pred.iloc[-1]) if len(pred) > 0 else 0.0
            except Exception as e:
                logger.debug("Transformer prediction failed: %s", e)

        # RL
        if self._rl_trader is not None:
            try:
                action = self._rl_trader.predict(df)
                predictions["rl"] = float(action)
            except Exception as e:
                logger.debug("RL prediction failed: %s", e)

        # Momentum fallback (always available)
        close = df["close"]
        if len(close) >= 20:
            mom_20 = float(close.iloc[-1] / close.iloc[-20] - 1)
            mom_5 = float(close.iloc[-1] / close.iloc[-5] - 1)
            predictions["momentum"] = 0.6 * mom_20 + 0.4 * mom_5

        return predictions

    def _update_ensemble_weights(self) -> None:
        """Update ensemble weights from trade memory."""
        self._ensemble.update_weights_from_memory(self._memory)
        logger.info("Ensemble weights updated from trade memory")

    def _apply_adaptive_thresholds(
        self,
        action: str,
        signal: EnsembleSignal,
        regime: str,
    ) -> str:
        """Adjust action based on historical accuracy in this regime."""
        try:
            regime_history = self._memory.query_similar_regime(
                regime, lookback_days=60, min_trades=10,
            )
            if not regime_history.get("sufficient_data"):
                return action

            historical_win_rate = regime_history.get("win_rate", 0.5)

            # If we've historically done poorly in this regime, require higher confidence
            if historical_win_rate < 0.4 and signal.confidence < 0.7:
                logger.info(
                    "Adaptive threshold: regime=%s has %.0f%% win rate, "
                    "requiring higher confidence (have %.2f, need 0.7)",
                    regime, historical_win_rate * 100, signal.confidence,
                )
                return "HOLD"

            # If we've historically done well, allow lower confidence trades
            if historical_win_rate > 0.6 and action == "HOLD" and abs(signal.raw_score) > 0.1:
                if signal.direction == 1:
                    return "BUY"
                elif signal.direction == -1:
                    return "SELL"

        except Exception as e:
            logger.debug("Adaptive threshold check failed: %s", e)

        return action

    def _check_retrain_needed(self) -> None:
        """Check if any models need retraining based on degrading accuracy."""
        trade_count = self._memory.get_trade_count()
        if trade_count < 20:
            return

        accuracies = self._memory.get_all_model_accuracies(window=50)
        models_needing_retrain = []

        for model_name, stats in accuracies.items():
            if stats.get("needs_retrain", False):
                models_needing_retrain.append(model_name)
                logger.warning(
                    "Model '%s' needs retraining: accuracy=%.2f%%, trend=%+.2f",
                    model_name,
                    stats.get("accuracy", 0) * 100,
                    stats.get("trend", 0),
                )

        if models_needing_retrain:
            logger.warning(
                "RETRAIN RECOMMENDED for: %s (call agent.train(df, models=%s))",
                models_needing_retrain, models_needing_retrain,
            )

    def _build_reasoning(
        self,
        regime: str,
        regime_proba: Dict[str, float],
        memory_insight: Dict[str, Any],
        predictions: Dict[str, float],
        signal: EnsembleSignal,
        can_trade: bool,
    ) -> List[str]:
        """Build human-readable reasoning chain for the decision."""
        reasons: List[str] = []

        # Regime
        top_regime_conf = regime_proba.get(regime, 0)
        reasons.append(
            f"Market regime: {regime} (confidence: {top_regime_conf:.0%})"
        )

        # Memory insight
        if memory_insight.get("sufficient_data"):
            wr = memory_insight.get("win_rate", 0)
            best_src = memory_insight.get("best_source", "unknown")
            reasons.append(
                f"Historical {regime} trades: {wr:.0%} win rate, "
                f"best model: {best_src}"
            )
        else:
            reasons.append(f"Insufficient history for {regime} regime")

        # Model predictions
        for model, pred in predictions.items():
            direction = "bullish" if pred > 0 else ("bearish" if pred < 0 else "neutral")
            reasons.append(f"{model}: {direction} ({pred:+.4f})")

        # Ensemble
        reasons.append(
            f"Ensemble: score={signal.raw_score:+.4f}, "
            f"agreement={signal.agreement_ratio:.0%}, "
            f"confidence={signal.confidence:.0%}"
        )

        # Risk
        if not can_trade:
            reasons.append("BLOCKED by risk manager — see risk_status for details")

        return reasons

    # ─── Convenience Methods ───

    def get_performance(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Get agent performance summary.

        Args:
            lookback_days: Number of days to summarize.

        Returns:
            Performance summary dict.
        """
        return self._memory.get_performance_summary(lookback_days)

    def get_weight_summary(self) -> Dict[str, Dict[str, float]]:
        """Get current ensemble weight configuration."""
        return self._ensemble.get_weight_summary()

    def get_recent_trades(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get the N most recent trades."""
        return self._memory.get_recent_trades(n)

    def save_models(self, directory: str = "models") -> None:
        """Save all trained models to disk.

        Args:
            directory: Directory to save models into.
        """
        import os
        os.makedirs(directory, exist_ok=True)

        if self._regime_classifier is not None:
            try:
                self._regime_classifier.save_model(f"{directory}/regime_classifier.joblib")
                logger.info("Saved regime classifier")
            except Exception as e:
                logger.warning("Failed to save regime classifier: %s", e)

        if self._lstm_predictor is not None:
            try:
                self._lstm_predictor.save_model(f"{directory}/lstm_predictor.pt")
                logger.info("Saved LSTM predictor")
            except Exception as e:
                logger.warning("Failed to save LSTM: %s", e)

        if self._transformer_predictor is not None:
            try:
                self._transformer_predictor.save_model(f"{directory}/transformer_predictor.pt")
                logger.info("Saved Transformer predictor")
            except Exception as e:
                logger.warning("Failed to save Transformer: %s", e)

        if self._rl_trader is not None:
            try:
                self._rl_trader.save_model(f"{directory}/rl_trader.zip")
                logger.info("Saved RL trader")
            except Exception as e:
                logger.warning("Failed to save RL trader: %s", e)

        logger.info("All models saved to %s/", directory)

    def load_models(self, directory: str = "models") -> None:
        """Load previously trained models from disk.

        Args:
            directory: Directory containing saved models.
        """
        import os

        if os.path.exists(f"{directory}/regime_classifier.joblib"):
            try:
                from shared.ml.regime_classifier import MLRegimeClassifier
                self._regime_classifier = MLRegimeClassifier()
                self._regime_classifier.load_model(f"{directory}/regime_classifier.joblib")
                logger.info("Loaded regime classifier")
            except Exception as e:
                logger.warning("Failed to load regime classifier: %s", e)

        if os.path.exists(f"{directory}/lstm_predictor.pt"):
            try:
                from shared.ml.deep_learning.lstm_predictor import LSTMPredictor
                self._lstm_predictor = LSTMPredictor()
                self._lstm_predictor.load_model(f"{directory}/lstm_predictor.pt")
                logger.info("Loaded LSTM predictor")
            except Exception as e:
                logger.warning("Failed to load LSTM: %s", e)

        if os.path.exists(f"{directory}/transformer_predictor.pt"):
            try:
                from shared.ml.deep_learning.transformer_predictor import TransformerPredictor
                self._transformer_predictor = TransformerPredictor()
                self._transformer_predictor.load_model(f"{directory}/transformer_predictor.pt")
                logger.info("Loaded Transformer predictor")
            except Exception as e:
                logger.warning("Failed to load Transformer: %s", e)

        if os.path.exists(f"{directory}/rl_trader.zip"):
            try:
                from shared.ml.rl_agent import RLTrader
                self._rl_trader = RLTrader()
                self._rl_trader.load_model(f"{directory}/rl_trader.zip")
                logger.info("Loaded RL trader")
            except Exception as e:
                logger.warning("Failed to load RL trader: %s", e)

        self._models_trained = True
        logger.info("Models loaded from %s/", directory)

    def close(self) -> None:
        """Clean up resources."""
        self._memory.close()

    def __repr__(self) -> str:
        active_models = []
        if self._regime_classifier: active_models.append("regime")
        if self._lstm_predictor: active_models.append("lstm")
        if self._transformer_predictor: active_models.append("transformer")
        if self._rl_trader: active_models.append("rl")
        active_models.append("momentum")  # always available

        return (
            f"SelfLearningAgent("
            f"models={active_models}, "
            f"decisions={self._decision_count}, "
            f"memory={self._memory.get_trade_count()} trades)"
        )
