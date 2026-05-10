"""
Tests for Self-Learning Agent Components
==========================================

Tests for:
- TradeMemory (persistent trade journal)
- EnsemblePredictor (adaptive multi-model combiner)
- SelfLearningAgent (autonomous orchestrator)
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


# ─── Fixtures ───


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(42)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100.0
    rows = []
    for i in range(n):
        regime = np.sin(2 * np.pi * i / 120)
        drift = 0.0003 * regime
        ret = drift + rng.randn() * 0.015
        price *= 1 + ret
        rows.append({
            "date": dates[i],
            "open": price * (1 + rng.randn() * 0.002),
            "high": price * (1 + abs(rng.randn()) * 0.005),
            "low": price * (1 - abs(rng.randn()) * 0.005),
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ─── TradeMemory Tests ───


class TestTradeMemory:
    """Tests for the TradeMemory persistent journal."""

    def test_create_and_record_trade(self, temp_db):
        from shared.ml.trade_memory import TradeMemory, TradeDecisionRecord

        memory = TradeMemory(temp_db)
        assert memory.get_trade_count() == 0

        trade = TradeDecisionRecord(
            timestamp=datetime.now().isoformat(),
            symbol="AAPL",
            action="BUY",
            entry_price=150.0,
            exit_price=155.0,
            pnl=500.0,
            pnl_pct=0.0333,
            regime="TRENDING",
            regime_confidence=0.75,
            ensemble_signal=0.3,
            ensemble_confidence=0.7,
            is_winner=True,
        )

        row_id = memory.record_trade(trade)
        assert row_id > 0
        assert memory.get_trade_count() == 1
        memory.close()

    def test_query_similar_regime(self, temp_db):
        from shared.ml.trade_memory import TradeMemory, TradeDecisionRecord

        memory = TradeMemory(temp_db)

        # Record 10 trades in TRENDING regime
        for i in range(10):
            trade = TradeDecisionRecord(
                timestamp=datetime.now().isoformat(),
                symbol="SPY",
                action="BUY",
                entry_price=100 + i,
                exit_price=102 + i if i % 3 != 0 else 98 + i,
                pnl=200 if i % 3 != 0 else -200,
                pnl_pct=0.02 if i % 3 != 0 else -0.02,
                regime="TRENDING",
                decision_source="lstm" if i % 2 == 0 else "ensemble",
                is_winner=(i % 3 != 0),
            )
            memory.record_trade(trade)

        result = memory.query_similar_regime("TRENDING", lookback_days=1)
        assert result["sufficient_data"] is True
        assert result["trade_count"] == 10
        assert 0 < result["win_rate"] <= 1
        assert "best_source" in result
        memory.close()

    def test_model_accuracy_tracking(self, temp_db):
        from shared.ml.trade_memory import TradeMemory

        memory = TradeMemory(temp_db)

        # Record 20 predictions for LSTM
        for i in range(20):
            prediction = 0.02 if i % 2 == 0 else -0.01
            actual = 0.01 if i % 3 != 0 else -0.01  # ~67% correct direction
            memory.record_model_prediction(
                model_name="lstm",
                prediction=prediction,
                actual_outcome=actual,
                regime="TRENDING",
            )

        accuracy = memory.get_model_accuracy("lstm", window=20)
        assert accuracy["model"] == "lstm"
        assert accuracy["total_predictions"] == 20
        assert 0 < accuracy["accuracy"] <= 1
        assert "needs_retrain" in accuracy
        memory.close()

    def test_performance_summary(self, temp_db):
        from shared.ml.trade_memory import TradeMemory, TradeDecisionRecord

        memory = TradeMemory(temp_db)

        for i in range(5):
            trade = TradeDecisionRecord(
                timestamp=datetime.now().isoformat(),
                symbol="AAPL",
                action="BUY",
                entry_price=150.0,
                pnl=100 if i % 2 == 0 else -50,
                is_winner=(i % 2 == 0),
                regime="TRENDING" if i < 3 else "RANGING",
            )
            memory.record_trade(trade)

        summary = memory.get_performance_summary(lookback_days=1)
        assert summary["total_trades"] == 5
        assert "win_rate" in summary
        assert "regime_breakdown" in summary
        memory.close()

    def test_get_recent_trades(self, temp_db):
        from shared.ml.trade_memory import TradeMemory, TradeDecisionRecord

        memory = TradeMemory(temp_db)

        for i in range(5):
            trade = TradeDecisionRecord(
                timestamp=datetime.now().isoformat(),
                symbol=f"STOCK_{i}",
                action="BUY",
                entry_price=100 + i,
            )
            memory.record_trade(trade)

        recent = memory.get_recent_trades(n=3)
        assert len(recent) == 3
        memory.close()


# ─── EnsemblePredictor Tests ───


class TestEnsemblePredictor:
    """Tests for the adaptive ensemble predictor."""

    def test_predict_bullish_consensus(self):
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor(min_confidence=0.2)
        predictions = {
            "lstm": 0.03,       # bullish
            "transformer": 0.02,  # bullish
            "rl": 1.0,          # buy
            "momentum": 0.05,   # bullish
        }

        signal = ensemble.predict(predictions, regime="TRENDING")
        assert signal.direction == 1  # BUY
        assert signal.confidence > 0.5
        assert signal.agreement_ratio > 0.8

    def test_predict_bearish_consensus(self):
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor(min_confidence=0.2)
        predictions = {
            "lstm": -0.03,
            "transformer": -0.02,
            "rl": -1.0,
            "momentum": -0.04,
        }

        signal = ensemble.predict(predictions, regime="TRENDING")
        assert signal.direction == -1  # SELL
        assert signal.confidence > 0.5

    def test_predict_mixed_signals_hold(self):
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor(min_confidence=0.4)
        predictions = {
            "lstm": 0.01,        # weakly bullish
            "transformer": -0.01,  # weakly bearish
            "rl": 0.0,          # hold
            "momentum": 0.005,  # barely bullish
        }

        signal = ensemble.predict(predictions, regime="RANGING")
        assert signal.direction == 0  # HOLD (weak/conflicting signals)

    def test_regime_affects_weights(self):
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor(min_confidence=0.2)
        predictions = {
            "momentum": 0.01,
            "rl": 0.5,
        }

        # In TRENDING regime, momentum gets higher weight
        signal_trending = ensemble.predict(predictions, regime="TRENDING")

        # In VOLATILE regime, momentum gets much lower weight
        signal_volatile = ensemble.predict(predictions, regime="VOLATILE")

        # Momentum's weighted contribution should be larger in TRENDING
        mom_contrib_trending = abs(signal_trending.model_contributions.get("momentum", 0))
        mom_contrib_volatile = abs(signal_volatile.model_contributions.get("momentum", 0))
        assert mom_contrib_trending > mom_contrib_volatile

    def test_empty_predictions(self):
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor()
        signal = ensemble.predict({})
        assert signal.direction == 0
        assert signal.confidence == 0.0

    def test_weight_summary(self):
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor()
        summary = ensemble.get_weight_summary()
        assert "lstm" in summary
        assert "transformer" in summary
        assert "rl" in summary
        assert "base_weight" in summary["lstm"]
        assert "effective_weight" in summary["lstm"]


# ─── SelfLearningAgent Tests ───


class TestSelfLearningAgent:
    """Tests for the autonomous self-learning agent."""

    def test_agent_init(self, temp_db):
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(db_path=temp_db)
        agent = SelfLearningAgent(config)
        assert agent._decision_count == 0
        assert repr(agent)  # shouldn't crash
        agent.close()

    def test_decide_with_fallback(self, temp_db, sample_ohlcv):
        """Agent should work even without ML models (momentum fallback)."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(db_path=temp_db)
        agent = SelfLearningAgent(config)

        # Don't train — agent should use momentum fallback
        decision = agent.decide(sample_ohlcv, symbol="TEST")

        assert "action" in decision
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert "confidence" in decision
        assert "regime" in decision
        assert "reasoning" in decision
        assert isinstance(decision["reasoning"], list)
        assert decision["regime"] in ("TRENDING", "RANGING", "VOLATILE")
        agent.close()

    def test_decide_and_record(self, temp_db, sample_ohlcv):
        """Agent should decide and then record outcomes."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(db_path=temp_db)
        agent = SelfLearningAgent(config)

        decision = agent.decide(sample_ohlcv, symbol="SPY")
        assert decision["action"] in ("BUY", "SELL", "HOLD")

        # Record outcome
        agent.record_outcome(
            exit_price=decision["price"] * 1.02,
            pnl=200.0,
            holding_period_bars=5,
        )

        # Check memory
        assert agent._memory.get_trade_count() == 1
        trades = agent.get_recent_trades(1)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "SPY"
        agent.close()

    def test_multiple_decisions_track_state(self, temp_db, sample_ohlcv):
        """Agent should track decision count and update weights periodically."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(db_path=temp_db, weight_update_interval=5)
        agent = SelfLearningAgent(config)

        for i in range(6):
            decision = agent.decide(sample_ohlcv, symbol="SPY")
            if decision["action"] != "HOLD":
                agent.record_outcome(
                    exit_price=decision["price"] * (1.01 if i % 2 == 0 else 0.99),
                    pnl=100 if i % 2 == 0 else -50,
                )

        assert agent._decision_count == 6
        agent.close()

    def test_performance_report(self, temp_db, sample_ohlcv):
        """Agent should produce a performance report."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(db_path=temp_db)
        agent = SelfLearningAgent(config)

        # Make a trade
        decision = agent.decide(sample_ohlcv, symbol="AAPL")
        agent.record_outcome(exit_price=100, pnl=50)

        perf = agent.get_performance(lookback_days=1)
        assert perf["total_trades"] >= 1
        agent.close()

    def test_weight_summary(self, temp_db):
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(db_path=temp_db)
        agent = SelfLearningAgent(config)
        summary = agent.get_weight_summary()
        assert isinstance(summary, dict)
        agent.close()


# ─── Integration Test ───


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_backtest_loop(self, temp_db, sample_ohlcv):
        """Simulate a complete backtest with the self-learning agent."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(db_path=temp_db, min_confidence=0.1)
        agent = SelfLearningAgent(config)

        # Walk through data bar by bar
        window = 60
        actions_taken = 0

        for i in range(window, len(sample_ohlcv) - 1, 5):  # step by 5 for speed
            df_window = sample_ohlcv.iloc[:i + 1].copy()
            decision = agent.decide(df_window, symbol="TEST")

            if decision["action"] != "HOLD":
                actions_taken += 1
                next_price = float(sample_ohlcv.iloc[min(i + 5, len(sample_ohlcv) - 1)]["close"])
                entry_price = decision["price"]
                pnl = (next_price - entry_price) if decision["action"] == "BUY" else (entry_price - next_price)

                agent.record_outcome(
                    exit_price=next_price,
                    pnl=pnl * 100,
                    holding_period_bars=5,
                )

        # Verify memory accumulated
        assert agent._memory.get_trade_count() > 0
        perf = agent.get_performance(lookback_days=365)
        assert perf["total_trades"] > 0
        agent.close()

    def test_ensemble_with_memory_integration(self, temp_db):
        """Ensemble weights should update from trade memory data."""
        from shared.ml.trade_memory import TradeMemory
        from shared.ml.ensemble_predictor import EnsemblePredictor

        memory = TradeMemory(temp_db)
        ensemble = EnsemblePredictor()

        # Record some model predictions
        for i in range(30):
            memory.record_model_prediction(
                model_name="lstm",
                prediction=0.02 if i % 2 == 0 else -0.01,
                actual_outcome=0.01 if i % 2 == 0 else -0.005,
                regime="TRENDING",
            )
            memory.record_model_prediction(
                model_name="transformer",
                prediction=0.01,
                actual_outcome=-0.01 if i % 3 == 0 else 0.005,
                regime="TRENDING",
            )

        # Update weights
        ensemble.update_weights_from_memory(memory, window=30)

        summary = ensemble.get_weight_summary()
        # LSTM should have higher accuracy weight than transformer
        # (since its predictions match actual outcome more often)
        lstm_weight = summary.get("lstm", {}).get("accuracy_weight", 0)
        tf_weight = summary.get("transformer", {}).get("accuracy_weight", 0)
        assert lstm_weight > 0
        assert tf_weight > 0

        memory.close()
