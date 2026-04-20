"""
Tests for shared/ml/self_learning_agent.py — expanded coverage
================================================================

Covers:
- decide() with trained/untrained model
- record_outcome() updates
- _check_retrain_needed() with degraded/healthy accuracy
- Auto-retrain with cooldown
- Adaptive threshold adjustment
- train() caches training data
"""
import os
import sys
import tempfile
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_ohlcv(n=300, seed=42, drift=0.0):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100.0
    rows = []
    for i in range(n):
        price *= 1 + drift + rng.randn() * 0.015
        rows.append({
            "date": dates[i],
            "open": price * 1.001,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": 1_000_000,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass  # Windows: file still locked by SQLite


@pytest.fixture
def agent(temp_db):
    from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig
    config = AgentConfig(
        db_path=temp_db,
        retrain_interval_trades=5,
        retrain_on_degradation=True,
        adaptive_thresholds=True,
        weight_update_interval=5,
    )
    a = SelfLearningAgent(config)
    # Keep reference to original memory for cleanup
    original_memory = a._memory
    yield a
    # Restore original memory before close if it was replaced by a mock
    a._memory = original_memory
    a.close()


@pytest.fixture
def sample_df():
    return _make_ohlcv(300, seed=42)


class TestDecideUntrained:
    def test_decide_without_training_returns_valid(self, agent, sample_df):
        decision = agent.decide(sample_df, symbol="AAPL")
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert 0 <= decision["confidence"] <= 1
        assert decision["regime"] in ("TRENDING", "RANGING", "VOLATILE")

    def test_decide_uses_momentum_fallback(self, agent, sample_df):
        decision = agent.decide(sample_df, symbol="SPY")
        assert "momentum" in decision["predictions"]

    def test_decide_increments_count(self, agent, sample_df):
        assert agent._decision_count == 0
        agent.decide(sample_df, symbol="X")
        assert agent._decision_count == 1
        agent.decide(sample_df, symbol="X")
        assert agent._decision_count == 2

    def test_decide_stores_pending_trade(self, agent, sample_df):
        agent.decide(sample_df, symbol="TEST")
        assert agent._pending_trade is not None
        assert agent._pending_trade["symbol"] == "TEST"
        assert "entry_price" in agent._pending_trade

    def test_decide_includes_reasoning(self, agent, sample_df):
        decision = agent.decide(sample_df)
        assert isinstance(decision["reasoning"], list)
        assert len(decision["reasoning"]) > 0


class TestDecideTrained:
    def test_decide_with_mock_regime(self, agent, sample_df):
        mock_rc = MagicMock()
        mock_regime = MagicMock()
        mock_regime.name = "TRENDING"
        mock_rc.predict.return_value = mock_regime
        mock_rc.predict_proba.return_value = {"TRENDING": 0.8, "RANGING": 0.1, "VOLATILE": 0.1}
        agent._regime_classifier = mock_rc
        agent._models_trained = True
        decision = agent.decide(sample_df, symbol="SPY")
        assert decision["regime"] == "TRENDING"

    def test_decide_with_mock_lstm(self, agent, sample_df):
        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = pd.Series([0.05])
        agent._lstm_predictor = mock_lstm
        decision = agent.decide(sample_df, symbol="SPY")
        assert "lstm" in decision["predictions"]
        assert decision["predictions"]["lstm"] == 0.05


class TestRecordOutcome:
    def test_record_clears_pending(self, agent, sample_df):
        agent.decide(sample_df, symbol="X")
        assert agent._pending_trade is not None
        agent.record_outcome(exit_price=105.0, pnl=500.0, holding_period_bars=5)
        assert agent._pending_trade is None

    def test_record_without_pending_warns(self, agent):
        agent.record_outcome(exit_price=100.0, pnl=0.0)

    def test_record_stores_in_memory(self, agent, sample_df):
        agent.decide(sample_df, symbol="SPY")
        agent.record_outcome(exit_price=105.0, pnl=500.0)
        assert agent._memory.get_trade_count() == 1

    def test_record_updates_risk_manager(self, agent, sample_df):
        agent.decide(sample_df, symbol="SPY")
        with patch.object(agent._risk_manager, "record_trade") as mock_record:
            agent.record_outcome(exit_price=105.0, pnl=500.0)
            mock_record.assert_called_once()


class TestCheckRetrainNeeded:
    def test_no_retrain_with_few_trades(self, agent):
        agent._check_retrain_needed()

    def test_retrain_triggered_on_degradation(self, agent, sample_df):
        agent._training_data = sample_df
        agent._last_retrain_time = 0.0
        mock_memory = MagicMock()
        mock_memory.get_trade_count.return_value = 100
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.30, "needs_retrain": True, "trend": -0.05},
        }
        agent._memory = mock_memory
        with patch.object(agent, "train") as mock_train:
            mock_train.return_value = {}
            agent._check_retrain_needed()
            mock_train.assert_called_once()
            assert "lstm" in mock_train.call_args[1]["models"]

    def test_retrain_respects_cooldown(self, agent, sample_df):
        agent._training_data = sample_df
        agent._last_retrain_time = time.time()
        mock_memory = MagicMock()
        mock_memory.get_trade_count.return_value = 100
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.30, "needs_retrain": True, "trend": -0.05},
        }
        agent._memory = mock_memory
        with patch.object(agent, "train") as mock_train:
            agent._check_retrain_needed()
            mock_train.assert_not_called()

    def test_retrain_healthy_accuracy_no_action(self, agent):
        mock_memory = MagicMock()
        mock_memory.get_trade_count.return_value = 100
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.65, "needs_retrain": False, "trend": 0.01},
        }
        agent._memory = mock_memory
        with patch.object(agent, "train") as mock_train:
            agent._check_retrain_needed()
            mock_train.assert_not_called()


class TestAdaptiveThresholds:
    def test_hold_on_low_win_rate(self, agent, sample_df):
        from shared.ml.ensemble_predictor import EnsembleSignal
        mock_memory = MagicMock()
        mock_memory.query_similar_regime.return_value = {
            "sufficient_data": True,
            "win_rate": 0.3,
        }
        agent._memory = mock_memory
        signal = EnsembleSignal(
            direction=1, raw_score=0.2, confidence=0.5,
            agreement_ratio=0.6, model_contributions={}, regime="TRENDING",
        )
        result = agent._apply_adaptive_thresholds("BUY", signal, "TRENDING")
        assert result == "HOLD"

    def test_allow_trade_on_high_win_rate(self, agent, sample_df):
        from shared.ml.ensemble_predictor import EnsembleSignal
        mock_memory = MagicMock()
        mock_memory.query_similar_regime.return_value = {
            "sufficient_data": True,
            "win_rate": 0.7,
        }
        agent._memory = mock_memory
        signal = EnsembleSignal(
            direction=1, raw_score=0.15, confidence=0.5,
            agreement_ratio=0.6, model_contributions={}, regime="TRENDING",
        )
        result = agent._apply_adaptive_thresholds("HOLD", signal, "TRENDING")
        assert result == "BUY"


class TestTrainCachesData:
    def test_train_caches_training_data(self, agent, sample_df):
        assert agent._training_data is None
        with patch("shared.ml.regime_classifier.MLRegimeClassifier") as mock_rc:
            mock_rc.return_value.fit.return_value = {"accuracy": 0.8}
            agent.train(sample_df, models=["regime"])
        assert agent._training_data is sample_df

    def test_train_sets_models_trained(self, agent, sample_df):
        assert agent._models_trained is False
        with patch("shared.ml.regime_classifier.MLRegimeClassifier") as mock_rc:
            mock_rc.return_value.fit.return_value = {"accuracy": 0.8}
            agent.train(sample_df, models=["regime"])
        assert agent._models_trained is True


class TestFallbackRegime:
    def test_fallback_returns_valid_regime(self, agent, sample_df):
        regime, proba = agent._fallback_regime(sample_df)
        assert regime in ("TRENDING", "RANGING", "VOLATILE")
        assert sum(proba.values()) == pytest.approx(1.0)


class TestGatherPredictions:
    def test_momentum_always_present(self, agent, sample_df):
        preds = agent._gather_predictions(sample_df)
        assert "momentum" in preds

    def test_lstm_prediction_included(self, agent, sample_df):
        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = pd.Series([0.02])
        agent._lstm_predictor = mock_lstm
        preds = agent._gather_predictions(sample_df)
        assert "lstm" in preds
        assert preds["lstm"] == 0.02

    def test_transformer_prediction_included(self, agent, sample_df):
        mock_tf = MagicMock()
        mock_tf.predict.return_value = pd.Series([0.03])
        agent._transformer_predictor = mock_tf
        preds = agent._gather_predictions(sample_df)
        assert "transformer" in preds

    def test_rl_prediction_included(self, agent, sample_df):
        mock_rl = MagicMock()
        mock_rl.predict.return_value = 1
        agent._rl_trader = mock_rl
        preds = agent._gather_predictions(sample_df)
        assert "rl" in preds
        assert preds["rl"] == 1.0

    def test_lstm_exception_handled(self, agent, sample_df):
        mock_lstm = MagicMock()
        mock_lstm.predict.side_effect = RuntimeError("fail")
        agent._lstm_predictor = mock_lstm
        preds = agent._gather_predictions(sample_df)
        assert "lstm" not in preds

    def test_transformer_exception_handled(self, agent, sample_df):
        mock_tf = MagicMock()
        mock_tf.predict.side_effect = RuntimeError("fail")
        agent._transformer_predictor = mock_tf
        preds = agent._gather_predictions(sample_df)
        assert "transformer" not in preds

    def test_rl_exception_handled(self, agent, sample_df):
        mock_rl = MagicMock()
        mock_rl.predict.side_effect = RuntimeError("fail")
        agent._rl_trader = mock_rl
        preds = agent._gather_predictions(sample_df)
        assert "rl" not in preds


class TestBuildReasoning:
    def test_reasoning_contains_regime(self, agent):
        from shared.ml.ensemble_predictor import EnsembleSignal
        signal = EnsembleSignal(
            direction=1, raw_score=0.2, confidence=0.7,
            agreement_ratio=0.8, model_contributions={"momentum": 0.2},
            regime="TRENDING",
        )
        reasons = agent._build_reasoning(
            "TRENDING", {"TRENDING": 0.8, "RANGING": 0.1, "VOLATILE": 0.1},
            {"sufficient_data": False}, {"momentum": 0.05}, signal, True,
        )
        assert any("TRENDING" in r for r in reasons)

    def test_reasoning_with_memory_insight(self, agent):
        from shared.ml.ensemble_predictor import EnsembleSignal
        signal = EnsembleSignal(
            direction=0, raw_score=0.0, confidence=0.3,
            agreement_ratio=0.5, model_contributions={},
            regime="RANGING",
        )
        reasons = agent._build_reasoning(
            "RANGING", {"RANGING": 0.7},
            {"sufficient_data": True, "win_rate": 0.6, "best_source": "lstm"},
            {"momentum": 0.01}, signal, True,
        )
        assert any("lstm" in r for r in reasons)

    def test_reasoning_risk_blocked(self, agent):
        from shared.ml.ensemble_predictor import EnsembleSignal
        signal = EnsembleSignal(
            direction=1, raw_score=0.3, confidence=0.8,
            agreement_ratio=0.9, model_contributions={},
            regime="TRENDING",
        )
        reasons = agent._build_reasoning(
            "TRENDING", {"TRENDING": 0.7}, {}, {"momentum": 0.05}, signal, False,
        )
        assert any("BLOCKED" in r for r in reasons)

    def test_reasoning_model_predictions(self, agent):
        from shared.ml.ensemble_predictor import EnsembleSignal
        signal = EnsembleSignal(
            direction=1, raw_score=0.3, confidence=0.8,
            agreement_ratio=0.9, model_contributions={},
            regime="TRENDING",
        )
        reasons = agent._build_reasoning(
            "TRENDING", {"TRENDING": 0.7}, {},
            {"lstm": 0.03, "momentum": -0.01}, signal, True,
        )
        assert any("lstm" in r and "bullish" in r for r in reasons)
        assert any("momentum" in r and "bearish" in r for r in reasons)


class TestUpdateEnsembleWeights:
    def test_updates_from_memory(self, agent):
        with patch.object(agent._ensemble, "update_weights_from_memory") as mock_update:
            agent._update_ensemble_weights()
            mock_update.assert_called_once_with(agent._memory)


class TestClassifyRegime:
    def test_with_classifier(self, agent, sample_df):
        mock_rc = MagicMock()
        mock_regime = MagicMock()
        mock_regime.name = "VOLATILE"
        mock_rc.predict.return_value = mock_regime
        mock_rc.predict_proba.return_value = {"VOLATILE": 0.9, "TRENDING": 0.05, "RANGING": 0.05}
        agent._regime_classifier = mock_rc
        regime, proba = agent._classify_regime(sample_df)
        assert regime == "VOLATILE"
        assert proba["VOLATILE"] == 0.9

    def test_classifier_exception_falls_back(self, agent, sample_df):
        mock_rc = MagicMock()
        mock_rc.predict.side_effect = RuntimeError("broken")
        agent._regime_classifier = mock_rc
        regime, proba = agent._classify_regime(sample_df)
        assert regime in ("TRENDING", "RANGING", "VOLATILE")


class TestTrainModels:
    def test_train_regime_with_mock(self, agent, sample_df):
        with patch("shared.ml.regime_classifier.MLRegimeClassifier") as mock_cls:
            mock_cls.return_value.fit.return_value = {"accuracy": 0.85}
            results = agent.train(sample_df, models=["regime"], verbose=False)
            assert "regime" in results
            assert results["regime"]["accuracy"] == 0.85

    def test_train_lstm_import_error(self, agent, sample_df):
        with patch.dict("sys.modules", {"shared.ml.deep_learning.lstm_predictor": None}):
            results = agent.train(sample_df, models=["lstm"], verbose=False)
            assert results["lstm"]["status"] == "skipped"

    def test_train_rl_import_error(self, agent, sample_df):
        with patch.dict("sys.modules", {"shared.ml.rl_agent": None}):
            results = agent.train(sample_df, models=["rl"], verbose=False)
            assert results["rl"]["status"] == "skipped"

    def test_train_transformer_import_error(self, agent, sample_df):
        with patch.dict("sys.modules", {"shared.ml.deep_learning.transformer_predictor": None}):
            results = agent.train(sample_df, models=["transformer"], verbose=False)
            assert results["transformer"]["status"] == "skipped"

    def test_train_lstm_with_mock(self, agent, sample_df):
        mock_lstm_module = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.train.return_value = {"val_loss": 0.001}
        mock_lstm_module.LSTMPredictor.return_value = mock_predictor
        mock_lstm_module.LSTMConfig.return_value = MagicMock()
        with patch.dict("sys.modules", {"shared.ml.deep_learning.lstm_predictor": mock_lstm_module}):
            results = agent.train(sample_df, models=["lstm"], verbose=True)
            assert "lstm" in results

    def test_train_transformer_with_mock(self, agent, sample_df):
        mock_tf_module = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.train.return_value = {"val_loss": 0.002}
        mock_tf_module.TransformerPredictor.return_value = mock_predictor
        mock_tf_module.TransformerConfig.return_value = MagicMock()
        with patch.dict("sys.modules", {"shared.ml.deep_learning.transformer_predictor": mock_tf_module}):
            results = agent.train(sample_df, models=["transformer"], verbose=True)
            assert "transformer" in results

    def test_train_rl_with_mock(self, agent, sample_df):
        mock_rl_module = MagicMock()
        mock_trader = MagicMock()
        mock_trader.train.return_value = {"total_timesteps": 50000}
        mock_rl_module.RLTrader.return_value = mock_trader
        with patch.dict("sys.modules", {"shared.ml.rl_agent": mock_rl_module}):
            results = agent.train(sample_df, models=["rl"], verbose=True)
            assert "rl" in results

    def test_train_default_models(self, agent, sample_df):
        with patch("shared.ml.regime_classifier.MLRegimeClassifier") as mock_cls:
            mock_cls.return_value.fit.return_value = {"accuracy": 0.8}
            results = agent.train(sample_df, verbose=False)
            assert "regime" in results


class TestSaveLoadModels:
    def test_save_with_no_models(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        agent.save_models(d)

    def test_save_with_regime_classifier(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        mock_rc = MagicMock()
        agent._regime_classifier = mock_rc
        agent.save_models(d)
        mock_rc.save_model.assert_called_once()

    def test_save_with_lstm(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        mock_lstm = MagicMock()
        agent._lstm_predictor = mock_lstm
        agent.save_models(d)
        mock_lstm.save_model.assert_called_once()

    def test_save_with_transformer(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        mock_tf = MagicMock()
        agent._transformer_predictor = mock_tf
        agent.save_models(d)
        mock_tf.save_model.assert_called_once()

    def test_save_with_rl(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        mock_rl = MagicMock()
        agent._rl_trader = mock_rl
        agent.save_models(d)
        mock_rl.save_model.assert_called_once()

    def test_save_exception_handled(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        mock_rc = MagicMock()
        mock_rc.save_model.side_effect = RuntimeError("disk full")
        agent._regime_classifier = mock_rc
        agent.save_models(d)

    def test_load_models_empty_dir(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        agent.load_models(d)
        assert agent._models_trained is True

    def test_load_regime_classifier(self, agent):
        import tempfile
        d = tempfile.mkdtemp()
        model_path = os.path.join(d, "regime_classifier.joblib")
        with open(model_path, "w") as f:
            f.write("dummy")
        with patch("shared.ml.regime_classifier.MLRegimeClassifier") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            agent.load_models(d)
            mock_instance.load_model.assert_called_once()

    def test_repr(self, agent):
        r = repr(agent)
        assert "SelfLearningAgent" in r
        assert "momentum" in r


class TestGetPerformance:
    def test_get_performance(self, agent):
        perf = agent.get_performance(lookback_days=7)
        assert isinstance(perf, dict)

    def test_get_weight_summary(self, agent):
        summary = agent.get_weight_summary()
        assert isinstance(summary, dict)

    def test_get_recent_trades(self, agent):
        trades = agent.get_recent_trades(5)
        assert isinstance(trades, list)
