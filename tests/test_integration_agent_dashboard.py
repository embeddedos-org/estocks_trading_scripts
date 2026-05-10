# -*- coding: utf-8 -*-
"""
Integration Test: Agent -> Graph -> Dashboard WebSocket Flow
==============================================================

End-to-end test verifying the full chain:
  SelfLearningAgent.decide() / record_outcome()
    -> GraphMemory.record_trade()
      -> on_change callback fires
        -> Dashboard WebSocket broadcasts to connected clients

Tests cover:
- Agent produces a decision and records an outcome
- GraphMemory receives the trade and fires observer callbacks
- Dashboard API reflects the updated graph state
- WebSocket endpoint accepts connections and receives broadcast messages
- Full round-trip: agent trade -> graph update -> WS client receives event
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

networkx_available = True
try:
    import networkx as nx
except ImportError:
    networkx_available = False

fastapi_available = True
try:
    from starlette.testclient import TestClient
except ImportError:
    fastapi_available = False

pytestmark = pytest.mark.skipif(
    not (networkx_available and fastapi_available),
    reason="networkx and fastapi/starlette required",
)


# ─── Fixtures ───


@pytest.fixture
def sample_ohlcv():
    """300-bar synthetic OHLCV data."""
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
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def graph_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ─── Test: Observer Callback Wiring ───


class TestObserverCallbackWiring:
    """Verify GraphMemory fires callbacks that can be collected externally."""

    def test_record_trade_fires_callback(self, graph_path):
        from shared.ml.graph_memory import GraphMemory

        events: List[tuple] = []
        gm = GraphMemory(path=graph_path, save_interval=100)
        gm.add_on_change(lambda e, d: events.append((e, d)))

        gm.record_trade({
            "regime": "TRENDING", "symbol": "AAPL", "action": "BUY",
            "pnl": 250, "decision_source": "ensemble",
        }, trade_id=1)

        assert len(events) == 1
        assert events[0][0] == "trade_recorded"
        assert events[0][1]["symbol"] == "AAPL"
        assert events[0][1]["pnl"] == 250.0
        gm.close()

    def test_regime_transition_fires_callback(self, graph_path):
        from shared.ml.graph_memory import GraphMemory

        events: List[tuple] = []
        gm = GraphMemory(path=graph_path, save_interval=100)
        gm.add_on_change(lambda e, d: events.append((e, d)))

        gm.record_trade({"regime": "TRENDING", "symbol": "AAPL", "pnl": 100}, trade_id=1)
        gm.record_trade({"regime": "VOLATILE", "symbol": "AAPL", "pnl": -50}, trade_id=2)

        event_types = [e for e, _ in events]
        assert "regime_transition" in event_types

        transition = next(d for e, d in events if e == "regime_transition")
        assert transition["from_regime"] == "TRENDING"
        assert transition["to_regime"] == "VOLATILE"
        gm.close()

    def test_multiple_callbacks(self, graph_path):
        from shared.ml.graph_memory import GraphMemory

        events_a: List[str] = []
        events_b: List[str] = []
        gm = GraphMemory(path=graph_path, save_interval=100)
        gm.add_on_change(lambda e, d: events_a.append(e))
        gm.add_on_change(lambda e, d: events_b.append(e))

        gm.record_trade({"regime": "TRENDING", "symbol": "SPY", "pnl": 50}, trade_id=1)

        assert events_a == ["trade_recorded"]
        assert events_b == ["trade_recorded"]
        gm.close()

    def test_callback_error_does_not_crash(self, graph_path):
        from shared.ml.graph_memory import GraphMemory

        def bad_callback(e, d):
            raise RuntimeError("intentional error")

        events: List[str] = []
        gm = GraphMemory(path=graph_path, save_interval=100)
        gm.add_on_change(bad_callback)
        gm.add_on_change(lambda e, d: events.append(e))

        gm.record_trade({"regime": "TRENDING", "symbol": "AAPL", "pnl": 100}, trade_id=1)

        # Second callback should still fire despite first one raising
        assert events == ["trade_recorded"]
        gm.close()


# ─── Test: Agent -> Graph Integration ───


class TestAgentGraphFlow:
    """Verify SelfLearningAgent.decide() + record_outcome() updates GraphMemory."""

    def test_decide_includes_graph_insight(self, sample_ohlcv, temp_db, graph_path):
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=temp_db,
            graph_memory_path=graph_path,
            use_graph_memory=True,
        )
        agent = SelfLearningAgent(config=config)
        assert agent._graph_memory is not None

        decision = agent.decide(sample_ohlcv, symbol="AAPL")
        assert "graph_insight" in decision
        agent.close()

    def test_record_outcome_populates_graph(self, sample_ohlcv, temp_db, graph_path):
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=temp_db,
            graph_memory_path=graph_path,
            use_graph_memory=True,
        )
        agent = SelfLearningAgent(config=config)

        decision = agent.decide(sample_ohlcv, symbol="AAPL")
        price = decision["price"]
        agent.record_outcome(exit_price=price * 1.02, pnl=200)

        gm = agent._graph_memory
        stats = gm.get_stats()
        assert stats["total_nodes"] > 0
        assert stats["node_types"].get("trade", 0) >= 1
        agent.close()

    def test_multiple_trades_build_graph(self, sample_ohlcv, temp_db, graph_path):
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=temp_db,
            graph_memory_path=graph_path,
            use_graph_memory=True,
        )
        agent = SelfLearningAgent(config=config)
        events: List[tuple] = []
        agent._graph_memory.add_on_change(lambda e, d: events.append((e, d)))

        for i in range(3):
            decision = agent.decide(sample_ohlcv, symbol="AAPL")
            pnl = 100 if i % 2 == 0 else -50
            agent.record_outcome(exit_price=decision["price"] * 1.01, pnl=pnl)

        trade_events = [e for e, _ in events if e == "trade_recorded"]
        assert len(trade_events) == 3

        stats = agent._graph_memory.get_stats()
        assert stats["node_types"].get("trade", 0) == 3
        agent.close()


# ─── Test: Dashboard API Reflects Graph State ───


class TestDashboardAPIReflectsGraph:
    """Verify the dashboard REST API returns data consistent with GraphMemory state."""

    def test_api_stats_after_trades(self, graph_path):
        from shared.ml.graph_memory import GraphMemory
        from shared.dashboard.app import app, configure

        gm = GraphMemory(path=graph_path, save_interval=100)
        for i in range(5):
            gm.record_trade({
                "regime": "TRENDING", "symbol": "AAPL", "action": "BUY",
                "pnl": 100, "is_winner": True, "decision_source": "ensemble",
            }, trade_id=i)
        gm.save()
        gm.close()

        configure(graph_path)
        client = TestClient(app)

        stats = client.get("/api/stats").json()
        assert stats["total_nodes"] > 0
        assert stats["node_types"]["trade"] == 5

        graph = client.get("/api/graph").json()
        trade_nodes = [n for n in graph["nodes"] if n["group"] == "trade"]
        assert len(trade_nodes) == 5

    def test_api_strategies_after_trades(self, graph_path):
        from shared.ml.graph_memory import GraphMemory
        from shared.dashboard.app import app, configure

        gm = GraphMemory(path=graph_path, save_interval=100)
        for i in range(3):
            gm.record_trade({
                "regime": "TRENDING", "symbol": "AAPL",
                "pnl": 200, "is_winner": True, "decision_source": "lstm",
            }, trade_id=i)
        gm.save()
        gm.close()

        configure(graph_path)
        client = TestClient(app)

        strats = client.get("/api/strategies").json()
        assert "TRENDING" in strats["strategies"]
        assert strats["strategies"]["TRENDING"]["strategy"] == "lstm"


# ─── Test: WebSocket Connection ───


class TestWebSocketConnection:
    """Verify the /ws WebSocket endpoint accepts and manages connections."""

    def test_websocket_connect_and_disconnect(self, graph_path):
        from shared.dashboard.app import app, configure, manager

        configure(graph_path)
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            assert manager.active_count >= 1

    def test_websocket_receives_broadcast(self, graph_path):
        """Directly broadcast a message via the manager and verify the WS client receives it."""
        import asyncio
        from shared.dashboard.app import app, configure, manager

        configure(graph_path)
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            # Broadcast from within the test's async context
            # TestClient runs the ASGI app in a thread, so we use the manager directly
            msg = {"type": "graph_update", "event": "trade_recorded", "data": {"trade_id": 99}}

            # Schedule broadcast on the app's event loop
            import threading

            def do_broadcast():
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(manager.broadcast(msg))
                loop.close()

            t = threading.Thread(target=do_broadcast)
            t.start()
            t.join(timeout=5)

            received = ws.receive_json(mode="text")
            assert received["type"] == "graph_update"
            assert received["event"] == "trade_recorded"
            assert received["data"]["trade_id"] == 99


# ─── Test: Full End-to-End Flow ───


class TestFullEndToEndFlow:
    """
    The complete integration chain:
    Agent.decide() -> Agent.record_outcome() -> GraphMemory.record_trade()
      -> on_change callback -> graph state updated -> Dashboard API reflects it
    """

    def test_agent_trade_updates_dashboard_api(self, sample_ohlcv, temp_db, graph_path):
        """Agent trades should be visible through the dashboard REST API."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig
        from shared.dashboard.app import app, configure

        config = AgentConfig(
            db_path=temp_db,
            graph_memory_path=graph_path,
            use_graph_memory=True,
        )
        agent = SelfLearningAgent(config=config)

        # Agent makes 3 decisions and records outcomes
        for i in range(3):
            decision = agent.decide(sample_ohlcv, symbol="SPY")
            pnl = 150 if i % 2 == 0 else -80
            agent.record_outcome(exit_price=decision["price"] * 1.01, pnl=pnl)

        agent._graph_memory.save()

        # Now configure the dashboard to read the same graph file
        configure(graph_path)
        client = TestClient(app)

        # Verify trades appear in the graph API
        graph = client.get("/api/graph").json()
        trade_nodes = [n for n in graph["nodes"] if n["group"] == "trade"]
        assert len(trade_nodes) == 3

        # Verify stats reflect the trades
        stats = client.get("/api/stats").json()
        assert stats["node_types"]["trade"] == 3
        assert stats["node_types"]["regime"] >= 1
        assert stats["node_types"]["symbol"] >= 1

        # Verify regime info appears
        assert stats["node_types"].get("strategy", 0) >= 1

        agent.close()

    def test_callback_captures_all_events(self, sample_ohlcv, temp_db, graph_path):
        """Observer callbacks should fire for every trade the agent records."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=temp_db,
            graph_memory_path=graph_path,
            use_graph_memory=True,
        )
        agent = SelfLearningAgent(config=config)

        captured: List[Dict[str, Any]] = []
        agent._graph_memory.add_on_change(
            lambda event, data: captured.append({"event": event, "data": data})
        )

        # Run 5 trade cycles
        for i in range(5):
            decision = agent.decide(sample_ohlcv, symbol="AAPL")
            pnl = 100 * (1 if i % 2 == 0 else -1)
            agent.record_outcome(exit_price=decision["price"] * 1.01, pnl=pnl)

        trade_events = [c for c in captured if c["event"] == "trade_recorded"]
        assert len(trade_events) == 5

        for te in trade_events:
            assert "trade_id" in te["data"]
            assert "symbol" in te["data"]
            assert te["data"]["symbol"] == "AAPL"
            assert "pnl" in te["data"]

        agent.close()

    def test_graph_persistence_survives_reload(self, sample_ohlcv, temp_db, graph_path):
        """Graph data written by the agent should survive a save/reload cycle."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig
        from shared.ml.graph_memory import GraphMemory

        config = AgentConfig(
            db_path=temp_db,
            graph_memory_path=graph_path,
            use_graph_memory=True,
        )
        agent = SelfLearningAgent(config=config)

        for i in range(3):
            decision = agent.decide(sample_ohlcv, symbol="TSLA")
            agent.record_outcome(exit_price=decision["price"] * 1.01, pnl=100)

        agent.close()  # triggers save

        # Reload from disk
        gm2 = GraphMemory(path=graph_path, save_interval=100)
        stats = gm2.get_stats()
        assert stats["node_types"].get("trade", 0) == 3
        assert stats["node_types"].get("symbol", 0) >= 1
        gm2.close()
