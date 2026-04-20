"""
Ensemble Predictor — Multi-Model Signal Combiner
====================================================

Combines predictions from multiple models (LSTM, Transformer, RL,
Regime Classifier) using adaptive weights that evolve based on each
model's recent accuracy.

Key features:
- Weighted voting across heterogeneous model types
- Adaptive weights: models that have been more accurate recently get more weight
- Regime-conditional weighting: different weight profiles per market regime
- Confidence scoring: higher agreement = higher confidence
- Integration with TradeMemory for historical accuracy lookup

Usage:
    ensemble = EnsemblePredictor()
    ensemble.update_weights_from_memory(trade_memory)
    signal = ensemble.predict(predictions_dict, regime="TRENDING")
    print(f"Signal: {signal.direction}, Confidence: {signal.confidence}")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal:
    """Output of the ensemble predictor."""

    direction: int  # -1 (SELL), 0 (HOLD), +1 (BUY)
    confidence: float  # 0.0 to 1.0
    raw_score: float  # weighted average before thresholding
    model_contributions: Dict[str, float]  # each model's weighted contribution
    agreement_ratio: float  # fraction of models that agree with final direction
    regime: str  # market regime used for weighting


@dataclass
class ModelWeight:
    """Weight configuration for a single model."""

    base_weight: float = 1.0
    accuracy_weight: float = 1.0  # derived from recent accuracy
    regime_multiplier: float = 1.0  # regime-specific adjustment

    @property
    def effective_weight(self) -> float:
        return self.base_weight * self.accuracy_weight * self.regime_multiplier


# Default base weights reflecting model strengths
_DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "lstm": 1.0,
    "transformer": 1.0,
    "rl": 0.8,
    "regime": 0.6,  # regime classifier used as a filter, not primary signal
    "momentum": 0.5,  # simple momentum fallback
    "sentiment": 0.7,  # news sentiment signal
}

# Regime-specific multipliers: which models shine in which conditions
_REGIME_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "TRENDING": {
        "lstm": 1.2,        # LSTM captures trends well
        "transformer": 1.1,  # attention helps with trend structure
        "rl": 1.3,          # RL excels at riding trends
        "regime": 0.8,
        "momentum": 1.4,    # momentum strategies love trends
        "sentiment": 1.0,   # sentiment confirms trend direction
    },
    "RANGING": {
        "lstm": 0.9,
        "transformer": 1.0,
        "rl": 0.7,          # RL tends to overtrade in ranges
        "regime": 1.2,      # regime detection important for mean reversion
        "momentum": 0.5,    # momentum fails in ranges
        "sentiment": 0.8,   # sentiment less useful in sideways markets
    },
    "VOLATILE": {
        "lstm": 0.7,        # high vol makes predictions noisy
        "transformer": 0.8,
        "rl": 0.5,          # RL can blow up in vol spikes
        "regime": 1.0,
        "momentum": 0.3,    # momentum is dangerous in volatility
        "sentiment": 1.3,   # news often drives vol — sentiment is key
    },
    "UNKNOWN": {
        "lstm": 1.0,
        "transformer": 1.0,
        "rl": 0.8,
        "regime": 0.6,
        "momentum": 0.5,
        "sentiment": 0.7,
    },
}


class EnsemblePredictor:
    """Adaptive multi-model ensemble with regime-conditional weighting.

    Args:
        base_weights: Override default base weights per model.
        buy_threshold: Minimum score to generate a BUY signal.
        sell_threshold: Maximum score to generate a SELL signal.
        min_confidence: Minimum confidence to act (below = HOLD).
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        buy_threshold: float = 0.15,
        sell_threshold: float = -0.15,
        min_confidence: float = 0.3,
    ) -> None:
        self._base_weights = base_weights or _DEFAULT_BASE_WEIGHTS.copy()
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._min_confidence = min_confidence
        self._regime_multipliers = copy.deepcopy(_REGIME_MULTIPLIERS)

        # Adaptive weights (updated from TradeMemory)
        self._model_weights: Dict[str, ModelWeight] = {
            name: ModelWeight(base_weight=w)
            for name, w in self._base_weights.items()
        }

        # Accuracy cache (updated periodically from TradeMemory)
        self._accuracy_cache: Dict[str, float] = {}

    def predict(
        self,
        predictions: Dict[str, float],
        regime: str = "UNKNOWN",
    ) -> EnsembleSignal:
        """Combine multiple model predictions into a single signal.

        Args:
            predictions: Dict mapping model_name to prediction value.
                Values should be normalized: positive = bullish, negative = bearish.
                Example: {"lstm": 0.02, "transformer": -0.01, "rl": 1, "momentum": 0.03}
            regime: Current market regime for weight adjustment.

        Returns:
            EnsembleSignal with direction, confidence, and breakdown.
        """
        if not predictions:
            return EnsembleSignal(
                direction=0, confidence=0.0, raw_score=0.0,
                model_contributions={}, agreement_ratio=0.0, regime=regime,
            )

        # Normalize predictions to [-1, +1] range
        normalized = self._normalize_predictions(predictions)

        # Apply regime-specific multipliers
        regime_mults = self._regime_multipliers.get(regime, self._regime_multipliers["UNKNOWN"])

        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0
        contributions: Dict[str, float] = {}

        for model_name, pred_value in normalized.items():
            weight = self._get_effective_weight(model_name, regime_mults)
            weighted_contribution = pred_value * weight
            weighted_sum += weighted_contribution
            total_weight += abs(weight)
            contributions[model_name] = weighted_contribution

        raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Agreement: what fraction of models agree on direction
        directions = [np.sign(v) for v in normalized.values() if v != 0]
        if directions:
            most_common = max(set(directions), key=directions.count)
            agreement = directions.count(most_common) / len(directions)
        else:
            agreement = 0.0

        # Confidence: based on signal strength and model agreement
        signal_strength = min(abs(raw_score) / 0.3, 1.0)  # normalize to [0, 1]
        confidence = 0.6 * signal_strength + 0.4 * agreement

        # Apply thresholds
        if confidence < self._min_confidence:
            direction = 0
        elif raw_score > self._buy_threshold:
            direction = 1
        elif raw_score < self._sell_threshold:
            direction = -1
        else:
            direction = 0

        signal = EnsembleSignal(
            direction=direction,
            confidence=round(confidence, 4),
            raw_score=round(raw_score, 6),
            model_contributions={k: round(v, 6) for k, v in contributions.items()},
            agreement_ratio=round(agreement, 4),
            regime=regime,
        )

        logger.debug(
            "Ensemble: direction=%+d, confidence=%.2f, raw=%.4f, agreement=%.2f, regime=%s",
            signal.direction, signal.confidence, signal.raw_score,
            signal.agreement_ratio, regime,
        )

        return signal

    def _normalize_predictions(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Normalize heterogeneous predictions to comparable [-1, +1] scale.

        Different models output different scales:
        - LSTM/Transformer: predicted return (e.g., 0.02 = +2%)
        - RL: discrete action (-1, 0, +1)
        - Regime: not a directional signal
        - Momentum: return percentage
        """
        normalized: Dict[str, float] = {}

        for name, value in predictions.items():
            if name == "rl":
                # Already in [-1, +1]
                normalized[name] = float(np.clip(value, -1, 1))
            elif name == "regime":
                # Regime isn't directional; treat as confidence modifier
                # Convert regime signal: TRENDING=+0.3, RANGING=0, VOLATILE=-0.3
                normalized[name] = float(np.clip(value, -1, 1))
            elif name in ("lstm", "transformer"):
                # Predicted returns: scale by ~50x to get to [-1, 1] range
                # A 2% predicted return → 1.0 signal
                normalized[name] = float(np.clip(value * 50, -1, 1))
            elif name == "momentum":
                # Momentum returns: scale similarly
                normalized[name] = float(np.clip(value * 30, -1, 1))
            elif name == "sentiment":
                # Sentiment score already in [-1, +1]
                normalized[name] = float(np.clip(value, -1, 1))
            else:
                # Generic: clip to [-1, 1]
                normalized[name] = float(np.clip(value, -1, 1))

        return normalized

    def _get_effective_weight(
        self,
        model_name: str,
        regime_mults: Dict[str, float],
    ) -> float:
        """Calculate effective weight for a model."""
        mw = self._model_weights.get(model_name)
        if mw is None:
            # Unknown model — use small default weight
            return 0.3

        mw.regime_multiplier = regime_mults.get(model_name, 1.0)
        return mw.effective_weight

    # ─── Adaptive Weight Updates ───

    def update_weights_from_memory(self, trade_memory: Any, window: int = 50) -> None:
        """Update model weights based on recent accuracy from TradeMemory.

        Models with higher recent accuracy get more weight.
        Models with degrading accuracy get less weight.

        Args:
            trade_memory: TradeMemory instance.
            window: Number of recent predictions to evaluate.
        """
        try:
            all_accuracies = trade_memory.get_all_model_accuracies(window)
        except Exception as e:
            logger.warning("Failed to fetch model accuracies: %s", e)
            return

        if not all_accuracies:
            logger.debug("No model accuracy data available yet")
            return

        for model_name, stats in all_accuracies.items():
            accuracy = stats.get("accuracy", 0.5)

            # Convert accuracy to weight multiplier:
            # 50% accuracy → 1.0 (baseline), 60% → 1.2, 40% → 0.8
            accuracy_weight = 0.4 + accuracy * 1.2

            # Penalize degrading models
            trend = stats.get("trend", 0.0)
            if trend < -0.1:
                accuracy_weight *= 0.8  # 20% penalty for degrading models

            if model_name in self._model_weights:
                self._model_weights[model_name].accuracy_weight = accuracy_weight
                logger.info(
                    "Weight updated: %s accuracy=%.2f%%, weight=%.3f (trend=%+.2f)",
                    model_name, accuracy * 100, accuracy_weight, trend,
                )
            else:
                self._model_weights[model_name] = ModelWeight(
                    base_weight=0.5,
                    accuracy_weight=accuracy_weight,
                )

        self._accuracy_cache = {
            k: v.get("accuracy", 0.5)
            for k, v in all_accuracies.items()
        }

        # GAP #3: Also update regime multipliers from trade memory
        self.update_regime_multipliers_from_memory(trade_memory)

    def update_regime_multipliers_from_memory(self, trade_memory: Any) -> None:
        """Update regime-specific model weights based on actual trade outcomes.

        Args:
            trade_memory: TradeMemory instance with get_model_accuracy(name, regime=regime).
        """
        for regime in ["TRENDING", "RANGING", "VOLATILE"]:
            for model_name in self._regime_multipliers.get(regime, {}):
                try:
                    accuracy = trade_memory.get_model_accuracy(model_name, regime=regime)
                    if accuracy is not None and accuracy.get("total_predictions", 0) >= 10:
                        acc_rate = accuracy["accuracy"]
                        # Scale multiplier: 0.5 at 40% accuracy, 1.0 at 50%, 1.5 at 60%
                        new_mult = max(0.3, min(2.0, acc_rate * 2.0))
                        self._regime_multipliers[regime][model_name] = new_mult
                        logger.info(
                            "Regime multiplier updated: %s/%s = %.2f (accuracy=%.1f%%)",
                            regime, model_name, new_mult, acc_rate * 100,
                        )
                except Exception as e:
                    logger.debug("Failed to update regime multiplier %s/%s: %s", regime, model_name, e)

    def save_weights(self, path: str) -> None:
        """Persist current model weights and regime multipliers to JSON.

        Args:
            path: File path to save weights JSON.
        """
        data = {
            "model_weights": {
                k: {"base": v.base_weight, "accuracy": v.accuracy_weight}
                for k, v in self._model_weights.items()
            },
            "regime_multipliers": self._regime_multipliers,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Ensemble weights saved to %s", path)

    def load_weights(self, path: str) -> None:
        """Load persisted weights from JSON.

        Args:
            path: File path to load weights from.
        """
        if not os.path.exists(path):
            logger.debug("No saved weights at %s", path)
            return
        with open(path) as f:
            data = json.load(f)
        for name, w in data.get("model_weights", {}).items():
            if name in self._model_weights:
                self._model_weights[name].accuracy_weight = w.get("accuracy", 1.0)
        loaded_mults = data.get("regime_multipliers", {})
        for regime, models in loaded_mults.items():
            if regime in self._regime_multipliers:
                self._regime_multipliers[regime].update(models)
        logger.info("Ensemble weights loaded from %s", path)

    def update_regime_weights(
        self,
        regime: str,
        model_name: str,
        multiplier: float,
    ) -> None:
        """Manually adjust regime-specific weight for a model.

        Args:
            regime: Market regime.
            model_name: Model to adjust.
            multiplier: New regime multiplier.
        """
        if regime in self._regime_multipliers:
            self._regime_multipliers[regime][model_name] = multiplier
            logger.info(
                "Regime weight updated: %s/%s = %.2f",
                regime, model_name, multiplier,
            )

    def get_weight_summary(self) -> Dict[str, Dict[str, float]]:
        """Get current weight configuration for all models.

        Returns:
            Dict mapping model_name to weight breakdown.
        """
        return {
            name: {
                "base_weight": mw.base_weight,
                "accuracy_weight": mw.accuracy_weight,
                "effective_weight": mw.effective_weight,
                "cached_accuracy": self._accuracy_cache.get(name, None),
            }
            for name, mw in self._model_weights.items()
        }

    def __repr__(self) -> str:
        models = list(self._model_weights.keys())
        return f"EnsemblePredictor(models={models}, buy_thresh={self._buy_threshold})"
