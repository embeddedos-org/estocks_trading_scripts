"""
Tests for shared.ml.ensemble_predictor — EnsemblePredictor
============================================================

Covers:
- predict() with various model combinations and regimes
- _normalize_predictions() scaling per model type
- _get_effective_weight() with regime multipliers
- update_weights_from_memory() adaptive weight updates
- update_regime_weights() manual regime weight adjustments
- get_weight_summary() weight introspection
- Instance-level dict isolation (not module-level)
- Weight normalization and regime transitions
- Edge cases: empty predictions, unknown models, unknown regimes
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared.ml.ensemble_predictor import (
    EnsemblePredictor,
    EnsembleSignal,
    ModelWeight,
    _DEFAULT_BASE_WEIGHTS,
    _REGIME_MULTIPLIERS,
)


# ─── Fixtures ───


@pytest.fixture
def predictor():
    """Fresh EnsemblePredictor with defaults."""
    return EnsemblePredictor()


@pytest.fixture
def custom_predictor():
    """Predictor with custom base weights and thresholds."""
    return EnsemblePredictor(
        base_weights={"modelA": 1.0, "modelB": 0.5},
        buy_threshold=0.10,
        sell_threshold=-0.10,
        min_confidence=0.2,
    )


# ─── ModelWeight dataclass ───


class TestModelWeight:
    def test_effective_weight_default(self):
        mw = ModelWeight()
        assert mw.effective_weight == 1.0

    def test_effective_weight_product(self):
        mw = ModelWeight(base_weight=2.0, accuracy_weight=0.5, regime_multiplier=1.5)
        assert mw.effective_weight == pytest.approx(1.5)

    def test_effective_weight_zero_component(self):
        mw = ModelWeight(base_weight=0.0, accuracy_weight=1.0, regime_multiplier=1.0)
        assert mw.effective_weight == 0.0


# ─── EnsembleSignal dataclass ───


class TestEnsembleSignal:
    def test_signal_attributes(self):
        sig = EnsembleSignal(
            direction=1, confidence=0.85, raw_score=0.25,
            model_contributions={"lstm": 0.15}, agreement_ratio=0.9, regime="TRENDING",
        )
        assert sig.direction == 1
        assert sig.confidence == 0.85
        assert sig.regime == "TRENDING"


# ─── predict() ───


class TestPredict:
    def test_empty_predictions_returns_hold(self, predictor):
        signal = predictor.predict({})
        assert signal.direction == 0
        assert signal.confidence == 0.0
        assert signal.raw_score == 0.0
        assert signal.model_contributions == {}
        assert signal.agreement_ratio == 0.0

    def test_strong_bullish_returns_buy(self, predictor):
        preds = {"lstm": 0.03, "transformer": 0.025, "rl": 1, "momentum": 0.05}
        signal = predictor.predict(preds, regime="TRENDING")
        assert signal.direction == 1
        assert signal.confidence > 0
        assert signal.raw_score > 0
        assert signal.regime == "TRENDING"

    def test_strong_bearish_returns_sell(self, predictor):
        preds = {"lstm": -0.03, "transformer": -0.025, "rl": -1, "momentum": -0.05}
        signal = predictor.predict(preds, regime="TRENDING")
        assert signal.direction == -1
        assert signal.raw_score < 0

    def test_mixed_signals_may_hold(self, predictor):
        preds = {"lstm": 0.001, "transformer": -0.001, "rl": 0, "momentum": 0.0}
        signal = predictor.predict(preds, regime="UNKNOWN")
        assert signal.direction == 0

    def test_regime_affects_weights(self, predictor):
        preds = {"lstm": 0.005, "momentum": 0.005}
        sig_trending = predictor.predict(preds, regime="TRENDING")
        sig_volatile = predictor.predict(preds, regime="VOLATILE")
        # Momentum has 1.4 in TRENDING vs 0.3 in VOLATILE — contributions differ
        assert sig_trending.model_contributions["momentum"] != sig_volatile.model_contributions["momentum"]

    def test_unknown_regime_uses_fallback(self, predictor):
        preds = {"lstm": 0.02}
        signal = predictor.predict(preds, regime="NONEXISTENT_REGIME")
        assert signal.regime == "NONEXISTENT_REGIME"
        # Should still produce a result using UNKNOWN multipliers
        assert isinstance(signal.direction, int)

    def test_agreement_ratio_all_agree(self, predictor):
        preds = {"lstm": 0.02, "transformer": 0.03, "rl": 1}
        signal = predictor.predict(preds, regime="UNKNOWN")
        assert signal.agreement_ratio == pytest.approx(1.0)

    def test_agreement_ratio_partial_agreement(self, predictor):
        preds = {"lstm": 0.02, "transformer": -0.03, "rl": 1}
        signal = predictor.predict(preds, regime="UNKNOWN")
        # 2 positive, 1 negative → 2/3
        assert signal.agreement_ratio == pytest.approx(2 / 3, abs=0.01)

    def test_model_contributions_present(self, predictor):
        preds = {"lstm": 0.02, "transformer": 0.01}
        signal = predictor.predict(preds, regime="UNKNOWN")
        assert "lstm" in signal.model_contributions
        assert "transformer" in signal.model_contributions

    def test_confidence_bounded_zero_to_one(self, predictor):
        for preds in [
            {"lstm": 0.05, "rl": 1, "momentum": 0.1},
            {"lstm": -0.001},
            {"rl": 0},
        ]:
            signal = predictor.predict(preds, regime="TRENDING")
            assert 0.0 <= signal.confidence <= 1.0

    def test_single_model_prediction(self, predictor):
        signal = predictor.predict({"sentiment": 0.8}, regime="VOLATILE")
        assert isinstance(signal.direction, int)
        assert signal.regime == "VOLATILE"

    def test_unknown_model_gets_small_weight(self, predictor):
        signal = predictor.predict({"alien_model": 0.5}, regime="UNKNOWN")
        # Unknown models get weight 0.3 — should still produce valid signal
        assert isinstance(signal, EnsembleSignal)

    def test_all_zero_predictions(self, predictor):
        preds = {"lstm": 0.0, "transformer": 0.0, "rl": 0}
        signal = predictor.predict(preds, regime="UNKNOWN")
        assert signal.direction == 0
        assert signal.raw_score == pytest.approx(0.0)
        assert signal.agreement_ratio == 0.0  # all zero → no direction


# ─── _normalize_predictions() ───


class TestNormalizePredictions:
    def test_lstm_scaled_by_50(self, predictor):
        result = predictor._normalize_predictions({"lstm": 0.02})
        assert result["lstm"] == pytest.approx(1.0)

    def test_lstm_clipped_to_range(self, predictor):
        result = predictor._normalize_predictions({"lstm": 0.1})
        assert result["lstm"] == pytest.approx(1.0)  # 0.1*50=5.0, clipped to 1.0

    def test_rl_already_normalized(self, predictor):
        result = predictor._normalize_predictions({"rl": -1})
        assert result["rl"] == -1.0

    def test_momentum_scaled_by_30(self, predictor):
        result = predictor._normalize_predictions({"momentum": 0.01})
        assert result["momentum"] == pytest.approx(0.3)

    def test_sentiment_passthrough(self, predictor):
        result = predictor._normalize_predictions({"sentiment": 0.5})
        assert result["sentiment"] == pytest.approx(0.5)

    def test_generic_model_clipped(self, predictor):
        result = predictor._normalize_predictions({"custom": 5.0})
        assert result["custom"] == pytest.approx(1.0)


# ─── Instance-level dict isolation (bug fix verification) ───


class TestInstanceIsolation:
    def test_separate_instances_have_independent_weights(self):
        """Verify fix: each instance has its own _regime_multipliers dict,
        not a shared module-level reference."""
        p1 = EnsemblePredictor()
        p2 = EnsemblePredictor()
        p1.update_regime_weights("TRENDING", "lstm", 99.0)
        # p2 should NOT see p1's change
        assert p2._regime_multipliers["TRENDING"]["lstm"] != 99.0
        assert p2._regime_multipliers["TRENDING"]["lstm"] == _REGIME_MULTIPLIERS["TRENDING"]["lstm"]

    def test_separate_instances_have_independent_model_weights(self):
        p1 = EnsemblePredictor()
        p2 = EnsemblePredictor()
        p1._model_weights["lstm"].accuracy_weight = 5.0
        assert p2._model_weights["lstm"].accuracy_weight == 1.0

    def test_custom_base_weights_dont_modify_defaults(self):
        custom = {"lstm": 99.0}
        p = EnsemblePredictor(base_weights=custom)
        assert _DEFAULT_BASE_WEIGHTS["lstm"] == 1.0
        assert p._base_weights["lstm"] == 99.0


# ─── update_regime_weights() ───


class TestUpdateRegimeWeights:
    def test_update_existing_regime_model(self, predictor):
        predictor.update_regime_weights("TRENDING", "lstm", 2.5)
        assert predictor._regime_multipliers["TRENDING"]["lstm"] == 2.5

    def test_update_nonexistent_regime_is_noop(self, predictor):
        predictor.update_regime_weights("FAKE_REGIME", "lstm", 2.5)
        # Should not crash or create new regime entry
        assert "FAKE_REGIME" not in predictor._regime_multipliers

    def test_update_new_model_in_existing_regime(self, predictor):
        predictor.update_regime_weights("TRENDING", "new_model", 0.9)
        assert predictor._regime_multipliers["TRENDING"]["new_model"] == 0.9

    def test_regime_weight_affects_prediction(self, predictor):
        preds = {"momentum": 0.005}  # small value so normalized doesn't saturate
        sig_before = predictor.predict(preds, regime="TRENDING")
        predictor.update_regime_weights("TRENDING", "momentum", 5.0)
        sig_after = predictor.predict(preds, regime="TRENDING")
        # Increasing momentum's regime weight should change contribution
        assert sig_after.model_contributions["momentum"] != sig_before.model_contributions["momentum"]


# ─── update_weights_from_memory() ───


class TestUpdateWeightsFromMemory:
    def test_updates_accuracy_weights(self, predictor):
        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.7, "trend": 0.05},
            "transformer": {"accuracy": 0.4, "trend": -0.2},
        }
        predictor.update_weights_from_memory(mock_memory, window=50)

        # lstm: 0.4 + 0.7*1.2 = 1.24
        assert predictor._model_weights["lstm"].accuracy_weight == pytest.approx(1.24)
        # transformer: 0.4 + 0.4*1.2 = 0.88, then *0.8 penalty = 0.704
        assert predictor._model_weights["transformer"].accuracy_weight == pytest.approx(0.704)

    def test_degrading_model_penalized(self, predictor):
        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.6, "trend": -0.15},
        }
        predictor.update_weights_from_memory(mock_memory)
        expected = (0.4 + 0.6 * 1.2) * 0.8  # degrading penalty
        assert predictor._model_weights["lstm"].accuracy_weight == pytest.approx(expected)

    def test_new_model_from_memory(self, predictor):
        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.return_value = {
            "new_fancy_model": {"accuracy": 0.65, "trend": 0.0},
        }
        predictor.update_weights_from_memory(mock_memory)
        assert "new_fancy_model" in predictor._model_weights
        assert predictor._model_weights["new_fancy_model"].base_weight == 0.5

    def test_empty_accuracies_is_noop(self, predictor):
        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.return_value = {}
        old_weights = {k: v.accuracy_weight for k, v in predictor._model_weights.items()}
        predictor.update_weights_from_memory(mock_memory)
        new_weights = {k: v.accuracy_weight for k, v in predictor._model_weights.items()}
        assert old_weights == new_weights

    def test_memory_exception_handled(self, predictor):
        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.side_effect = RuntimeError("DB error")
        # Should not raise
        predictor.update_weights_from_memory(mock_memory)

    def test_accuracy_cache_updated(self, predictor):
        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.72, "trend": 0.0},
        }
        predictor.update_weights_from_memory(mock_memory)
        assert predictor._accuracy_cache["lstm"] == 0.72


# ─── get_weight_summary() ───


class TestGetWeightSummary:
    def test_summary_contains_all_models(self, predictor):
        summary = predictor.get_weight_summary()
        for name in _DEFAULT_BASE_WEIGHTS:
            assert name in summary
            assert "base_weight" in summary[name]
            assert "accuracy_weight" in summary[name]
            assert "effective_weight" in summary[name]
            assert "cached_accuracy" in summary[name]

    def test_summary_accuracy_none_by_default(self, predictor):
        summary = predictor.get_weight_summary()
        assert summary["lstm"]["cached_accuracy"] is None

    def test_summary_after_memory_update(self, predictor):
        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.65, "trend": 0.0},
        }
        predictor.update_weights_from_memory(mock_memory)
        summary = predictor.get_weight_summary()
        assert summary["lstm"]["cached_accuracy"] == 0.65


# ─── repr ───


class TestRepr:
    def test_repr_format(self, predictor):
        r = repr(predictor)
        assert "EnsemblePredictor" in r
        assert "buy_thresh" in r

    def test_custom_predictor_repr(self, custom_predictor):
        r = repr(custom_predictor)
        assert "0.1" in r  # buy_threshold


# ─── Regime transition scenarios ───


class TestRegimeTransitions:
    def test_same_preds_different_regimes_different_signals(self, predictor):
        preds = {"lstm": 0.015, "transformer": 0.01, "rl": 1, "momentum": 0.03, "sentiment": 0.5}
        signals = {}
        for regime in ["TRENDING", "RANGING", "VOLATILE", "UNKNOWN"]:
            signals[regime] = predictor.predict(preds, regime=regime)
        # At minimum raw_scores should differ across regimes
        raw_scores = [s.raw_score for s in signals.values()]
        assert len(set(raw_scores)) > 1, "Expected different scores for different regimes"
