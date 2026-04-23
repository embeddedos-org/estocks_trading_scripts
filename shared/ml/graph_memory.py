# -*- coding: utf-8 -*-
"""
Graph Memory -- NetworkX-Based Relational Trade Memory
========================================================

Supplements the flat SQLite TradeMemory with a directed graph that captures
relationships between regimes, symbols, trades, strategies, and feature states.

Enables queries like:
- "When TRENDING transitions to VOLATILE, which strategy minimises drawdown?"
- "What do correlated symbols suggest about the next regime?"
- "In similar feature conditions, what was the best action?"

Node types: REGIME, SYMBOL, TRADE, STRATEGY, FEATURE_STATE
Edge types: TRANSITIONS_TO, TRADED_IN, TRADED_SYMBOL, USED_STRATEGY,
            SIMILAR_TO, CORRELATES_WITH, PRODUCED

Persistence: JSON via ``nx.node_link_data`` / ``nx.node_link_graph``.

Usage:
    gm = GraphMemory("graph_memory.json")
    gm.record_trade(trade_dict, trade_id=1)
    insight = gm.get_graph_enhanced_insight("TRENDING", "AAPL", features)
    gm.close()
"""

from __future__ import annotations

import json
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]


# ─── Enums ───


class NodeType(str, Enum):
    REGIME = "regime"
    SYMBOL = "symbol"
    TRADE = "trade"
    STRATEGY = "strategy"
    FEATURE_STATE = "feature_state"


class EdgeType(str, Enum):
    TRANSITIONS_TO = "transitions_to"
    TRADED_IN = "traded_in"
    TRADED_SYMBOL = "traded_symbol"
    USED_STRATEGY = "used_strategy"
    SIMILAR_TO = "similar_to"
    CORRELATES_WITH = "correlates_with"
    PRODUCED = "produced"


# ─── Helpers ───


def _node_id(node_type: NodeType, key: str) -> str:
    """Deterministic node identifier, e.g. ``regime:TRENDING``."""
    return f"{node_type.value}:{key}"


def _discretize_features(features: Dict[str, float]) -> str:
    """Bucketise continuous features into categorical states.

    Each feature is mapped to low / medium / high based on simple
    quantile-like thresholds derived from the value's sign and magnitude.
    The resulting string is a deterministic key for a FEATURE_STATE node.
    """
    parts: List[str] = []
    for name in sorted(features.keys()):
        val = features[name]
        if val < -0.5:
            bucket = "low"
        elif val > 0.5:
            bucket = "high"
        else:
            bucket = "medium"
        parts.append(f"{name}={bucket}")
    return "|".join(parts) if parts else "empty"


# ─── GraphMemory ───


class GraphMemory:
    """NetworkX-backed graph that models relationships between market entities.

    Args:
        path: File path for JSON persistence.
        save_interval: Auto-save after this many mutations.
    """

    def __init__(self, path: str = "graph_memory.json", save_interval: int = 10) -> None:
        if nx is None:
            raise ImportError(
                "networkx is required for GraphMemory. "
                "Install it with: pip install networkx>=3.0"
            )

        self._path = Path(path)
        self._save_interval = save_interval
        self._lock = threading.Lock()
        self._mutation_count = 0
        self._last_regime: Optional[str] = None
        self._on_change_callbacks: List[Callable[[str, Dict], None]] = []

        self._graph: nx.DiGraph = self._load()

        logger.info(
            "GraphMemory loaded: %d nodes, %d edges (path=%s)",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            self._path,
        )

    # ─── Persistence ───

    def _load(self) -> nx.DiGraph:
        """Load graph from JSON or create an empty DiGraph."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                return nx.node_link_graph(data, directed=True)
            except Exception as e:
                logger.warning("Corrupt graph file %s — starting fresh: %s", self._path, e)
        return nx.DiGraph()

    def save(self) -> None:
        """Persist the graph to JSON."""
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                data = nx.node_link_data(self._graph)
                with open(self._path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                logger.debug("Graph saved: %d nodes, %d edges", self._graph.number_of_nodes(), self._graph.number_of_edges())
            except Exception as e:
                logger.error("Failed to save graph: %s", e)

    def _maybe_autosave(self) -> None:
        """Auto-save every ``save_interval`` mutations."""
        self._mutation_count += 1
        if self._mutation_count >= self._save_interval:
            self.save()
            self._mutation_count = 0

    def close(self) -> None:
        """Persist final state and release resources."""
        self.save()

    def add_on_change(self, callback: Callable[[str, Dict], None]) -> None:
        """Register a callback invoked after each graph mutation.

        Args:
            callback: ``fn(event_type, data)`` called with a string event
                      type and a dict payload describing the change.
        """
        self._on_change_callbacks.append(callback)

    def _notify_change(self, event_type: str, data: Dict[str, Any]) -> None:
        """Fire all registered on-change callbacks (best-effort)."""
        for cb in self._on_change_callbacks:
            try:
                cb(event_type, data)
            except Exception as exc:
                logger.warning("on_change callback error: %s", exc)

    # ─── Graph Construction ───

    def record_trade(self, trade: Dict[str, Any], trade_id: int) -> None:
        """Record a completed trade and build graph edges.

        Creates/updates REGIME, SYMBOL, TRADE, STRATEGY, and FEATURE_STATE
        nodes with appropriate edges.

        Args:
            trade: Dict with keys: regime, symbol, action, pnl, pnl_pct,
                   ensemble_confidence, decision_source, features_snapshot, is_winner.
            trade_id: Unique numeric identifier for this trade.
        """
        with self._lock:
            g = self._graph
            regime = trade.get("regime", "UNKNOWN")
            symbol = trade.get("symbol", "UNKNOWN")
            action = trade.get("action", "HOLD")
            strategy = trade.get("decision_source", "ensemble")
            pnl = float(trade.get("pnl", 0))
            pnl_pct = float(trade.get("pnl_pct", 0))
            is_winner = bool(trade.get("is_winner", pnl > 0))
            confidence = float(trade.get("ensemble_confidence", 0))

            # Parse features
            features_raw = trade.get("features_snapshot", "{}")
            if isinstance(features_raw, str):
                try:
                    features = json.loads(features_raw)
                except (json.JSONDecodeError, TypeError):
                    features = {}
            else:
                features = features_raw if isinstance(features_raw, dict) else {}

            # ── Nodes ──
            regime_id = _node_id(NodeType.REGIME, regime)
            symbol_id = _node_id(NodeType.SYMBOL, symbol)
            trade_node_id = _node_id(NodeType.TRADE, str(trade_id))
            strategy_id = _node_id(NodeType.STRATEGY, strategy)

            # Regime node
            if not g.has_node(regime_id):
                g.add_node(regime_id, type=NodeType.REGIME.value, label=regime, trade_count=0, total_pnl=0.0, wins=0)
            g.nodes[regime_id]["trade_count"] += 1
            g.nodes[regime_id]["total_pnl"] += pnl
            if is_winner:
                g.nodes[regime_id]["wins"] += 1

            # Symbol node
            if not g.has_node(symbol_id):
                g.add_node(symbol_id, type=NodeType.SYMBOL.value, label=symbol, trade_count=0, total_pnl=0.0, wins=0)
            g.nodes[symbol_id]["trade_count"] += 1
            g.nodes[symbol_id]["total_pnl"] += pnl
            if is_winner:
                g.nodes[symbol_id]["wins"] += 1

            # Trade node
            g.add_node(
                trade_node_id,
                type=NodeType.TRADE.value,
                label=f"T{trade_id}",
                action=action,
                pnl=pnl,
                pnl_pct=pnl_pct,
                is_winner=is_winner,
                confidence=confidence,
            )

            # Strategy node
            if not g.has_node(strategy_id):
                g.add_node(strategy_id, type=NodeType.STRATEGY.value, label=strategy, trade_count=0, total_pnl=0.0, wins=0)
            g.nodes[strategy_id]["trade_count"] += 1
            g.nodes[strategy_id]["total_pnl"] += pnl
            if is_winner:
                g.nodes[strategy_id]["wins"] += 1

            # Feature state node
            feature_key = _discretize_features(features) if features else "empty"
            feature_id = _node_id(NodeType.FEATURE_STATE, feature_key)
            if not g.has_node(feature_id):
                g.add_node(feature_id, type=NodeType.FEATURE_STATE.value, label=feature_key, trade_count=0, wins=0)
            g.nodes[feature_id]["trade_count"] += 1
            if is_winner:
                g.nodes[feature_id]["wins"] += 1

            # ── Edges ──
            # Trade → Regime
            g.add_edge(trade_node_id, regime_id, type=EdgeType.TRADED_IN.value)
            # Trade → Symbol
            g.add_edge(trade_node_id, symbol_id, type=EdgeType.TRADED_SYMBOL.value)
            # Trade → Strategy
            g.add_edge(trade_node_id, strategy_id, type=EdgeType.USED_STRATEGY.value)
            # Strategy → Regime (PRODUCED edge with outcome stats)
            prod_key = (strategy_id, regime_id)
            if g.has_edge(*prod_key) and g.edges[prod_key].get("type") == EdgeType.PRODUCED.value:
                edge = g.edges[prod_key]
                edge["trade_count"] = edge.get("trade_count", 0) + 1
                edge["total_pnl"] = edge.get("total_pnl", 0.0) + pnl
                edge["wins"] = edge.get("wins", 0) + (1 if is_winner else 0)
                tc = edge["trade_count"]
                edge["win_rate"] = edge["wins"] / tc if tc > 0 else 0
                edge["avg_pnl"] = edge["total_pnl"] / tc if tc > 0 else 0
            else:
                g.add_edge(
                    strategy_id, regime_id,
                    type=EdgeType.PRODUCED.value,
                    trade_count=1,
                    total_pnl=pnl,
                    wins=1 if is_winner else 0,
                    win_rate=1.0 if is_winner else 0.0,
                    avg_pnl=pnl,
                )

            # Feature state → Regime (CORRELATES_WITH)
            corr_key = (feature_id, regime_id)
            if g.has_edge(*corr_key) and g.edges[corr_key].get("type") == EdgeType.CORRELATES_WITH.value:
                edge = g.edges[corr_key]
                edge["count"] = edge.get("count", 0) + 1
            else:
                g.add_edge(feature_id, regime_id, type=EdgeType.CORRELATES_WITH.value, count=1)

            # Regime transition tracking
            if self._last_regime is not None and self._last_regime != regime:
                self.record_regime_transition(self._last_regime, regime)
            self._last_regime = regime

        self._maybe_autosave()
        self._notify_change("trade_recorded", {
            "trade_id": trade_id,
            "regime": regime,
            "symbol": symbol,
            "action": action,
            "pnl": pnl,
            "is_winner": is_winner,
        })

    def record_regime_transition(
        self,
        from_regime: str,
        to_regime: str,
        duration_bars: int = 0,
    ) -> None:
        """Record a regime-to-regime transition.

        Args:
            from_regime: Regime being exited.
            to_regime: Regime being entered.
            duration_bars: How many bars the previous regime lasted.
        """
        from_id = _node_id(NodeType.REGIME, from_regime)
        to_id = _node_id(NodeType.REGIME, to_regime)

        g = self._graph

        # Ensure regime nodes exist
        for nid, label in ((from_id, from_regime), (to_id, to_regime)):
            if not g.has_node(nid):
                g.add_node(nid, type=NodeType.REGIME.value, label=label, trade_count=0, total_pnl=0.0, wins=0)

        edge_key = (from_id, to_id)
        if g.has_edge(*edge_key) and g.edges[edge_key].get("type") == EdgeType.TRANSITIONS_TO.value:
            edge = g.edges[edge_key]
            edge["count"] = edge.get("count", 0) + 1
            if duration_bars > 0:
                old_avg = edge.get("avg_duration_bars", 0)
                old_count = edge["count"] - 1
                edge["avg_duration_bars"] = (old_avg * old_count + duration_bars) / edge["count"]
        else:
            g.add_edge(
                from_id, to_id,
                type=EdgeType.TRANSITIONS_TO.value,
                count=1,
                avg_duration_bars=float(duration_bars),
            )

        self._notify_change("regime_transition", {
            "from_regime": from_regime,
            "to_regime": to_regime,
        })

    def update_symbol_correlations(self, correlation_matrix: Dict[str, Dict[str, float]]) -> None:
        """Update SIMILAR_TO edges between symbol nodes.

        Args:
            correlation_matrix: Nested dict mapping ``{sym_a: {sym_b: corr, ...}, ...}``.
                Only correlations with absolute value ≥ 0.5 are stored.
        """
        with self._lock:
            g = self._graph

            # Remove old SIMILAR_TO edges
            old_edges = [
                (u, v) for u, v, d in g.edges(data=True)
                if d.get("type") == EdgeType.SIMILAR_TO.value
            ]
            g.remove_edges_from(old_edges)

            # Add new
            for sym_a, correlations in correlation_matrix.items():
                a_id = _node_id(NodeType.SYMBOL, sym_a)
                if not g.has_node(a_id):
                    g.add_node(a_id, type=NodeType.SYMBOL.value, label=sym_a, trade_count=0, total_pnl=0.0, wins=0)
                for sym_b, corr in correlations.items():
                    if sym_a == sym_b:
                        continue
                    if abs(corr) < 0.5:
                        continue
                    b_id = _node_id(NodeType.SYMBOL, sym_b)
                    if not g.has_node(b_id):
                        g.add_node(b_id, type=NodeType.SYMBOL.value, label=sym_b, trade_count=0, total_pnl=0.0, wins=0)
                    g.add_edge(a_id, b_id, type=EdgeType.SIMILAR_TO.value, correlation=corr)
                    g.add_edge(b_id, a_id, type=EdgeType.SIMILAR_TO.value, correlation=corr)

        self._maybe_autosave()
        self._notify_change("correlations_updated", {
            "symbols": list(correlation_matrix.keys()),
        })

    # ─── Query Methods ───

    def get_regime_transition_probs(self, regime: str) -> Dict[str, float]:
        """Return transition probabilities from the given regime.

        Uses Markov-chain counts on TRANSITIONS_TO edges.

        Returns:
            Dict mapping next-regime → probability. Empty if no transitions recorded.
        """
        regime_id = _node_id(NodeType.REGIME, regime)
        g = self._graph

        if not g.has_node(regime_id):
            return {}

        transitions: Dict[str, int] = {}
        total = 0
        for _, target, data in g.out_edges(regime_id, data=True):
            if data.get("type") == EdgeType.TRANSITIONS_TO.value:
                target_label = g.nodes[target].get("label", target)
                count = data.get("count", 1)
                transitions[target_label] = count
                total += count

        if total == 0:
            return {}

        return {regime: count / total for regime, count in transitions.items()}

    def get_best_strategy_for_regime(self, regime: str) -> Optional[Dict[str, Any]]:
        """Find the strategy with the best score in a given regime.

        Score = win_rate × avg_pnl (so positive-EV strategies rank highest).

        Returns:
            Dict with strategy, win_rate, avg_pnl, trade_count, score — or None.
        """
        regime_id = _node_id(NodeType.REGIME, regime)
        g = self._graph

        if not g.has_node(regime_id):
            return None

        best: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        for source, _, data in g.in_edges(regime_id, data=True):
            if data.get("type") != EdgeType.PRODUCED.value:
                continue
            if g.nodes[source].get("type") != NodeType.STRATEGY.value:
                continue
            win_rate = data.get("win_rate", 0)
            avg_pnl = data.get("avg_pnl", 0)
            trade_count = data.get("trade_count", 0)
            score = win_rate * avg_pnl
            if score > best_score:
                best_score = score
                best = {
                    "strategy": g.nodes[source].get("label", source),
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "trade_count": trade_count,
                    "score": score,
                }

        return best

    def get_correlated_symbols(self, symbol: str, min_corr: float = 0.5) -> List[Dict[str, Any]]:
        """Find symbols connected via SIMILAR_TO edges.

        Args:
            symbol: Ticker to query neighbours for.
            min_corr: Minimum absolute correlation threshold.

        Returns:
            List of dicts with symbol and correlation.
        """
        symbol_id = _node_id(NodeType.SYMBOL, symbol)
        g = self._graph

        if not g.has_node(symbol_id):
            return []

        results: List[Dict[str, Any]] = []
        for _, target, data in g.out_edges(symbol_id, data=True):
            if data.get("type") != EdgeType.SIMILAR_TO.value:
                continue
            corr = data.get("correlation", 0)
            if abs(corr) >= min_corr:
                results.append({
                    "symbol": g.nodes[target].get("label", target),
                    "correlation": corr,
                })

        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return results

    def get_similar_conditions(
        self,
        regime: str,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Find trades recorded under similar feature-state + regime conditions.

        Traverses FEATURE_STATE → CORRELATES_WITH → REGIME path.

        Returns:
            Dict with matching_trades, win_rate, avg_pnl.
        """
        feature_key = _discretize_features(features) if features else "empty"
        feature_id = _node_id(NodeType.FEATURE_STATE, feature_key)
        regime_id = _node_id(NodeType.REGIME, regime)
        g = self._graph

        result: Dict[str, Any] = {"matching_trades": 0, "win_rate": 0.0, "avg_pnl": 0.0}

        if not g.has_node(feature_id):
            return result

        # Check if this feature state has been seen with this regime
        if g.has_edge(feature_id, regime_id):
            edge = g.edges[(feature_id, regime_id)]
            if edge.get("type") == EdgeType.CORRELATES_WITH.value:
                count = edge.get("count", 0)
                result["matching_trades"] = count

        # Aggregate stats from trades in this regime via feature state
        node_data = g.nodes.get(feature_id, {})
        tc = node_data.get("trade_count", 0)
        wins = node_data.get("wins", 0)
        if tc > 0:
            result["matching_trades"] = tc
            result["win_rate"] = wins / tc

        # Get avg_pnl from regime node
        regime_data = g.nodes.get(regime_id, {})
        rtc = regime_data.get("trade_count", 0)
        if rtc > 0:
            result["avg_pnl"] = regime_data.get("total_pnl", 0) / rtc

        return result

    def get_graph_enhanced_insight(
        self,
        regime: str,
        symbol: str,
        features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Composite query combining all graph intelligence.

        Returns a dict suitable for the SelfLearningAgent's decide() method.

        Args:
            regime: Current market regime.
            symbol: Current ticker.
            features: Feature dict (continuous values, will be discretised).

        Returns:
            Dict with regime_transition, best_strategy, correlated_symbols_signal,
            condition_match, graph_confidence, sufficient_data.
        """
        features = features or {}

        # Count trade nodes for data-sufficiency check
        trade_node_count = sum(
            1 for _, d in self._graph.nodes(data=True)
            if d.get("type") == NodeType.TRADE.value
        )
        sufficient = trade_node_count >= 10

        # 1. Regime transition prediction
        transition_probs = self.get_regime_transition_probs(regime)
        regime_transition: Dict[str, Any] = {}
        if transition_probs:
            next_regime = max(transition_probs, key=transition_probs.get)  # type: ignore[arg-type]
            regime_transition = {
                "next_regime": next_regime,
                "probability": transition_probs[next_regime],
                "all_probs": transition_probs,
            }

        # 2. Best strategy
        best_strat = self.get_best_strategy_for_regime(regime)

        # 3. Correlated symbols signal
        corr_symbols = self.get_correlated_symbols(symbol)
        corr_signal: Dict[str, Any] = {"agreement": 0.0, "details": corr_symbols}
        if corr_symbols:
            # Compute agreement: weighted average of correlated symbols' win rates
            total_weight = 0.0
            weighted_wins = 0.0
            for cs in corr_symbols:
                cs_id = _node_id(NodeType.SYMBOL, cs["symbol"])
                cs_data = self._graph.nodes.get(cs_id, {})
                cs_tc = cs_data.get("trade_count", 0)
                cs_wins = cs_data.get("wins", 0)
                if cs_tc > 0:
                    w = abs(cs["correlation"])
                    weighted_wins += w * (cs_wins / cs_tc)
                    total_weight += w
            if total_weight > 0:
                corr_signal["agreement"] = weighted_wins / total_weight

        # 4. Condition match
        condition_match = self.get_similar_conditions(regime, features)

        # 5. Graph confidence — scales with data density
        node_count = self._graph.number_of_nodes()
        edge_count = self._graph.number_of_edges()
        density_score = min(1.0, (node_count + edge_count) / 200)
        data_quality = min(1.0, trade_node_count / 50) if trade_node_count > 0 else 0.0
        graph_confidence = 0.5 * density_score + 0.5 * data_quality

        return {
            "regime_transition": regime_transition,
            "best_strategy": best_strat,
            "correlated_symbols_signal": corr_signal,
            "condition_match": condition_match,
            "graph_confidence": round(graph_confidence, 4),
            "sufficient_data": sufficient,
        }

    # ─── Diagnostics ───

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the graph."""
        g = self._graph
        type_counts: Dict[str, int] = {}
        for _, d in g.nodes(data=True):
            nt = d.get("type", "unknown")
            type_counts[nt] = type_counts.get(nt, 0) + 1

        edge_type_counts: Dict[str, int] = {}
        for _, _, d in g.edges(data=True):
            et = d.get("type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        return {
            "total_nodes": g.number_of_nodes(),
            "total_edges": g.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
        }

    def __repr__(self) -> str:
        return (
            f"GraphMemory(nodes={self._graph.number_of_nodes()}, "
            f"edges={self._graph.number_of_edges()}, "
            f"path={self._path})"
        )
