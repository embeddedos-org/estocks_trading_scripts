"""
Tests for GraphMemory — NetworkX-Based Relational Trade Memory
================================================================

Unit tests:
- Empty graph creation
- Single trade recording (node + edge creation)
- Multiple trades in same regime (attribute updates)
- Regime transition tracking
- Transition probability correctness
- Best strategy ranking
- Symbol correlation edges
- Similar condition traversal
- Composite insight (sparse/dense)
- JSON persistence round-trip

Integration tests:
- Agent with graph memory enabled
- Agent with graph memory disabled
- Graceful degradation when networkx is missing
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict

import pytest

networkx_available = True
try:
    import networkx as nx
except ImportError:
    networkx_available = False

pytestmark = pytest.mark.skipif(not networkx_available, reason="networkx not installed")


# ─── Helpers ───


def _make_trade(
    regime: str = "TRENDING",
    symbol: str = "AAPL",
    action: str = "BUY",
    pnl: float = 100.0,
    pnl_pct: float = 0.02,
    is_winner: bool = True,
    decision_source: str = "ensemble",
    ensemble_confidence: float = 0.75,
    features: Dict[str, float] | None = None,
) -> Dict:
    feats = features or {"rsi_14": 0.6, "adx_14": 0.8, "volatility": -0.3}
    return {
        "regime": regime,
        "symbol": symbol,
        "action": action,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "is_winner": is_winner,
        "decision_source": decision_source,
        "ensemble_confidence": ensemble_confidence,
        "features_snapshot": json.dumps(feats),
    }


@pytest.fixture
def graph_path():
    """Temporary file for graph persistence."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    # Remove so GraphMemory starts fresh
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def gm(graph_path):
    """Fresh GraphMemory instance."""
    from shared.ml.graph_memory import GraphMemory
    mem = GraphMemory(path=graph_path, save_interval=100)
    yield mem
    mem.close()


# ─── Unit Tests ───


class TestGraphMemoryCreation:
    def test_create_empty_graph(self, gm):
        """Empty GraphMemory should have 0 nodes and 0 edges."""
        assert gm._graph.number_of_nodes() == 0
        assert gm._graph.number_of_edges() == 0

    def test_repr(self, gm):
        r = repr(gm)
        assert "GraphMemory" in r
        assert "nodes=0" in r


class TestRecordTrade:
    def test_record_single_trade(self, gm):
        """Recording one trade should create 5 node types and edges."""
        gm.record_trade(_make_trade(), trade_id=1)

        g = gm._graph
        node_types = {d.get("type") for _, d in g.nodes(data=True)}
        assert "regime" in node_types
        assert "symbol" in node_types
        assert "trade" in node_types
        assert "strategy" in node_types
        assert "feature_state" in node_types

        # Should have edges: trade→regime, trade→symbol, trade→strategy, strategy→regime, feature→regime
        assert g.number_of_edges() >= 5

    def test_record_multiple_trades_same_regime(self, gm):
        """Multiple trades update regime node attributes."""
        for i in range(5):
            gm.record_trade(
                _make_trade(pnl=100 if i % 2 == 0 else -50, is_winner=i % 2 == 0),
                trade_id=i,
            )

        from shared.ml.graph_memory import _node_id, NodeType
        regime_id = _node_id(NodeType.REGIME, "TRENDING")
        data = gm._graph.nodes[regime_id]
        assert data["trade_count"] == 5
        assert data["wins"] == 3  # trades 0, 2, 4

    def test_trade_node_attributes(self, gm):
        """Trade node should carry action, pnl, is_winner."""
        gm.record_trade(_make_trade(action="SELL", pnl=-200, is_winner=False), trade_id=42)

        from shared.ml.graph_memory import _node_id, NodeType
        trade_id = _node_id(NodeType.TRADE, "42")
        data = gm._graph.nodes[trade_id]
        assert data["action"] == "SELL"
        assert data["pnl"] == -200
        assert data["is_winner"] is False


class TestRegimeTransitions:
    def test_regime_transition_tracking(self, gm):
        """Switching regimes should create TRANSITIONS_TO edges."""
        gm.record_trade(_make_trade(regime="TRENDING"), trade_id=1)
        gm.record_trade(_make_trade(regime="VOLATILE"), trade_id=2)
        gm.record_trade(_make_trade(regime="RANGING"), trade_id=3)

        from shared.ml.graph_memory import _node_id, NodeType, EdgeType
        t_id = _node_id(NodeType.REGIME, "TRENDING")
        v_id = _node_id(NodeType.REGIME, "VOLATILE")
        assert gm._graph.has_edge(t_id, v_id)
        assert gm._graph.edges[(t_id, v_id)]["type"] == EdgeType.TRANSITIONS_TO.value

    def test_regime_transition_probs(self, gm):
        """Transition probabilities should sum to ~1.0."""
        # TRENDING → VOLATILE ×3, TRENDING → RANGING ×1
        gm.record_trade(_make_trade(regime="TRENDING"), trade_id=1)
        gm.record_trade(_make_trade(regime="VOLATILE"), trade_id=2)
        gm.record_trade(_make_trade(regime="TRENDING"), trade_id=3)
        gm.record_trade(_make_trade(regime="VOLATILE"), trade_id=4)
        gm.record_trade(_make_trade(regime="TRENDING"), trade_id=5)
        gm.record_trade(_make_trade(regime="VOLATILE"), trade_id=6)
        gm.record_trade(_make_trade(regime="TRENDING"), trade_id=7)
        gm.record_trade(_make_trade(regime="RANGING"), trade_id=8)

        probs = gm.get_regime_transition_probs("TRENDING")
        assert len(probs) >= 2
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert probs.get("VOLATILE", 0) > probs.get("RANGING", 0)

    def test_transition_probs_empty(self, gm):
        """Non-existent regime should return empty dict."""
        assert gm.get_regime_transition_probs("NONEXISTENT") == {}


class TestBestStrategy:
    def test_best_strategy_for_regime(self, gm):
        """Strategy with best win_rate × avg_pnl should rank first."""
        # "ensemble" strategy with good results
        for i in range(5):
            gm.record_trade(
                _make_trade(decision_source="ensemble", pnl=200, is_winner=True),
                trade_id=i,
            )
        # "lstm" strategy with bad results
        for i in range(5, 10):
            gm.record_trade(
                _make_trade(decision_source="lstm", pnl=-100, is_winner=False),
                trade_id=i,
            )

        best = gm.get_best_strategy_for_regime("TRENDING")
        assert best is not None
        assert best["strategy"] == "ensemble"
        assert best["win_rate"] == 1.0
        assert best["score"] > 0

    def test_best_strategy_none(self, gm):
        """No data should return None."""
        assert gm.get_best_strategy_for_regime("UNKNOWN") is None


class TestCorrelatedSymbols:
    def test_update_correlations(self, gm):
        """SIMILAR_TO edges should be created for correlated symbols."""
        corr = {
            "AAPL": {"MSFT": 0.85, "TSLA": 0.3},
            "MSFT": {"AAPL": 0.85, "TSLA": 0.2},
            "TSLA": {"AAPL": 0.3, "MSFT": 0.2},
        }
        gm.update_symbol_correlations(corr)

        result = gm.get_correlated_symbols("AAPL")
        symbols = [r["symbol"] for r in result]
        assert "MSFT" in symbols
        assert "TSLA" not in symbols  # corr < 0.5

    def test_correlated_symbols_empty(self, gm):
        """Non-existent symbol should return empty list."""
        assert gm.get_correlated_symbols("NONEXISTENT") == []


class TestSimilarConditions:
    def test_similar_conditions(self, gm):
        """Should find trades under similar feature + regime conditions."""
        features = {"rsi_14": 0.6, "adx_14": 0.8, "volatility": -0.3}
        for i in range(5):
            gm.record_trade(
                _make_trade(features=features, is_winner=i < 3),
                trade_id=i,
            )

        result = gm.get_similar_conditions("TRENDING", features)
        assert result["matching_trades"] > 0
        assert 0 <= result["win_rate"] <= 1.0

    def test_similar_conditions_no_data(self, gm):
        """No matching data should return zero counts."""
        result = gm.get_similar_conditions("TRENDING", {"rsi_14": 0.6})
        assert result["matching_trades"] == 0


class TestCompositeInsight:
    def test_graph_enhanced_insight_sparse(self, gm):
        """With < 10 trades, sufficient_data should be False."""
        for i in range(5):
            gm.record_trade(_make_trade(), trade_id=i)

        insight = gm.get_graph_enhanced_insight("TRENDING", "AAPL")
        assert insight["sufficient_data"] is False
        assert "graph_confidence" in insight

    def test_graph_enhanced_insight_dense(self, gm):
        """With >= 10 trades, sufficient_data should be True and all keys present."""
        for i in range(15):
            gm.record_trade(
                _make_trade(
                    regime="TRENDING" if i < 10 else "VOLATILE",
                    pnl=100 if i % 2 == 0 else -50,
                    is_winner=i % 2 == 0,
                ),
                trade_id=i,
            )

        insight = gm.get_graph_enhanced_insight("TRENDING", "AAPL", {"rsi_14": 0.6})
        assert insight["sufficient_data"] is True
        assert "regime_transition" in insight
        assert "best_strategy" in insight
        assert "correlated_symbols_signal" in insight
        assert "condition_match" in insight
        assert 0 <= insight["graph_confidence"] <= 1.0


class TestPersistence:
    def test_save_and_load(self, graph_path):
        """Saved graph should be identical after reload."""
        from shared.ml.graph_memory import GraphMemory

        gm1 = GraphMemory(path=graph_path, save_interval=100)
        for i in range(5):
            gm1.record_trade(_make_trade(regime="TRENDING"), trade_id=i)
        gm1.save()

        n1 = gm1._graph.number_of_nodes()
        e1 = gm1._graph.number_of_edges()

        gm2 = GraphMemory(path=graph_path, save_interval=100)
        assert gm2._graph.number_of_nodes() == n1
        assert gm2._graph.number_of_edges() == e1

        gm1.close()
        gm2.close()

    def test_corrupt_json_recovery(self, graph_path):
        """Corrupt JSON file should result in fresh empty graph."""
        from shared.ml.graph_memory import GraphMemory

        with open(graph_path, "w") as f:
            f.write("{invalid json!!!}")

        gm = GraphMemory(path=graph_path, save_interval=100)
        assert gm._graph.number_of_nodes() == 0
        gm.close()

    def test_autosave(self, graph_path):
        """Graph should auto-save after save_interval mutations."""
        from shared.ml.graph_memory import GraphMemory

        gm = GraphMemory(path=graph_path, save_interval=3)
        for i in range(4):
            gm.record_trade(_make_trade(), trade_id=i)

        # After 3+ mutations, file should exist and have content
        assert os.path.exists(graph_path)
        with open(graph_path) as f:
            data = json.load(f)
        assert "nodes" in data or "links" in data
        gm.close()


class TestStats:
    def test_get_stats(self, gm):
        """Stats should report correct node/edge type counts."""
        gm.record_trade(_make_trade(), trade_id=1)
        stats = gm.get_stats()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert "regime" in stats["node_types"]
        assert "trade" in stats["node_types"]


class TestHelpers:
    def test_node_id(self):
        from shared.ml.graph_memory import _node_id, NodeType
        assert _node_id(NodeType.REGIME, "TRENDING") == "regime:TRENDING"
        assert _node_id(NodeType.SYMBOL, "AAPL") == "symbol:AAPL"

    def test_discretize_features(self):
        from shared.ml.graph_memory import _discretize_features
        result = _discretize_features({"a": -1.0, "b": 0.0, "c": 1.0})
        assert "a=low" in result
        assert "b=medium" in result
        assert "c=high" in result

    def test_discretize_empty(self):
        from shared.ml.graph_memory import _discretize_features
        assert _discretize_features({}) == "empty"


# ─── Integration Tests ───


class TestAgentIntegration:
    @pytest.fixture
    def sample_ohlcv(self):
        """Synthetic OHLCV data."""
        import numpy as np
        import pandas as pd

        rng = np.random.RandomState(42)
        n = 300
        dates = pd.bdate_range("2020-01-01", periods=n)
        price = 100.0
        rows = []
        for i in range(n):
            ret = rng.randn() * 0.015
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
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_agent_with_graph_memory(self, sample_ohlcv, temp_db, graph_path):
        """Full decide/record_outcome cycle with graph memory."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=temp_db,
            graph_memory_path=graph_path,
            use_graph_memory=True,
            graph_save_interval=5,
        )
        agent = SelfLearningAgent(config=config)
        assert agent._graph_memory is not None

        decision = agent.decide(sample_ohlcv, symbol="TEST")
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert "graph_insight" in decision

        agent.record_outcome(exit_price=decision["price"] * 1.01, pnl=100)
        agent.close()

        # Graph file should exist after close
        assert os.path.exists(graph_path)

    def test_agent_graph_memory_disabled(self, sample_ohlcv, temp_db, graph_path):
        """Agent should work normally with graph memory disabled."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=temp_db,
            use_graph_memory=False,
        )
        agent = SelfLearningAgent(config=config)
        assert agent._graph_memory is None

        decision = agent.decide(sample_ohlcv, symbol="TEST")
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert decision.get("graph_insight") is None
        agent.close()

    def test_agent_networkx_missing(self, sample_ohlcv, temp_db, monkeypatch):
        """Agent should degrade gracefully when networkx import fails."""
        import shared.ml.self_learning_agent as sla_module

        # Patch the import inside __init__ to fail
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "shared.ml.graph_memory":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=temp_db,
            use_graph_memory=True,
        )

        monkeypatch.setattr("builtins.__import__", mock_import)
        try:
            agent = SelfLearningAgent(config=config)
            assert agent._graph_memory is None
        finally:
            monkeypatch.undo()

        agent.close()
