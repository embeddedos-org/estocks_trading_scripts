# -*- coding: utf-8 -*-
"""
Graph Memory Dashboard -- FastAPI Backend
==========================================

Serves an interactive web dashboard for visualising the GraphMemory graph.
Includes WebSocket support for real-time graph updates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.ml.graph_memory import EdgeType, GraphMemory, NodeType, _node_id

app = FastAPI(title="Graphify Memory Dashboard")

_gm: GraphMemory | None = None

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

NODE_STYLE = {
    NodeType.REGIME.value: {"color": "#e74c3c", "shape": "diamond", "size": 30},
    NodeType.SYMBOL.value: {"color": "#3498db", "shape": "dot", "size": 25},
    NodeType.STRATEGY.value: {"color": "#9b59b6", "shape": "triangle", "size": 22},
    NodeType.TRADE.value: {"color": "#2ecc71", "shape": "dot", "size": 12},
    NodeType.FEATURE_STATE.value: {"color": "#95a5a6", "shape": "square", "size": 15},
}

EDGE_STYLE = {
    EdgeType.TRANSITIONS_TO.value: {"color": "#e74c3c", "dashes": False, "width": 2, "arrows": "to"},
    EdgeType.PRODUCED.value: {"color": "#9b59b6", "dashes": False, "width": 1, "arrows": "to"},
    EdgeType.TRADED_IN.value: {"color": "#7f8c8d", "dashes": [5, 5], "width": 1, "arrows": "to"},
    EdgeType.TRADED_SYMBOL.value: {"color": "#3498db", "dashes": [5, 5], "width": 1, "arrows": "to"},
    EdgeType.USED_STRATEGY.value: {"color": "#8e44ad", "dashes": [5, 5], "width": 1, "arrows": "to"},
    EdgeType.SIMILAR_TO.value: {"color": "#f39c12", "dashes": [2, 4], "width": 1, "arrows": ""},
    EdgeType.CORRELATES_WITH.value: {"color": "#95a5a6", "dashes": [2, 4], "width": 1, "arrows": "to"},
}


# ─── WebSocket Connection Manager ───


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        stale: List[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()
_event_loop: asyncio.AbstractEventLoop | None = None


def _on_graph_change(event_type: str, data: Dict[str, Any]) -> None:
    """Bridge sync GraphMemory callback → async WebSocket broadcast.

    ``record_trade()`` runs in a sync thread, so we use
    ``asyncio.run_coroutine_threadsafe()`` to schedule the broadcast
    on the running event loop.
    """
    if _event_loop is None or _event_loop.is_closed():
        return
    message = {"type": "graph_update", "event": event_type, "data": data}
    asyncio.run_coroutine_threadsafe(manager.broadcast(message), _event_loop)


# ─── File-watch background task ───

_graph_path: str = ""
_last_mtime: float = 0.0


async def _file_watch_loop() -> None:
    """Periodically check if the graph JSON was modified externally."""
    global _last_mtime
    while True:
        await asyncio.sleep(2)
        try:
            if _graph_path and os.path.exists(_graph_path):
                mtime = os.path.getmtime(_graph_path)
                if _last_mtime > 0 and mtime > _last_mtime:
                    await manager.broadcast({
                        "type": "graph_update",
                        "event": "file_changed",
                        "data": {"path": _graph_path},
                    })
                _last_mtime = mtime
        except Exception:
            pass


@app.on_event("startup")
async def _startup() -> None:
    global _event_loop, _last_mtime
    _event_loop = asyncio.get_running_loop()
    if _graph_path and os.path.exists(_graph_path):
        _last_mtime = os.path.getmtime(_graph_path)
    asyncio.create_task(_file_watch_loop())


# ─── Configuration ───


def configure(graph_path: str) -> None:
    """Load the GraphMemory instance from disk."""
    global _gm, _graph_path, _last_mtime
    _graph_path = graph_path
    _gm = GraphMemory(path=graph_path)
    _gm.add_on_change(_on_graph_change)
    if os.path.exists(graph_path):
        _last_mtime = os.path.getmtime(graph_path)


def _gm_or_error() -> GraphMemory:
    if _gm is None:
        raise HTTPException(status_code=503, detail="GraphMemory not loaded")
    return _gm


# ─── HTTP Endpoints ───


@app.get("/", response_class=HTMLResponse)
async def index():
    template = TEMPLATE_DIR / "index.html"
    if not template.exists():
        raise HTTPException(status_code=500, detail="Template not found")
    return HTMLResponse(content=template.read_text(encoding="utf-8"))


@app.get("/api/graph")
async def get_graph() -> Dict[str, Any]:
    """Return vis.js-compatible nodes + edges."""
    gm = _gm_or_error()
    g = gm._graph

    nodes: List[Dict[str, Any]] = []
    for nid, data in g.nodes(data=True):
        ntype = data.get("type", "unknown")
        style = NODE_STYLE.get(ntype, {"color": "#ffffff", "shape": "dot", "size": 10})

        color = style["color"]
        if ntype == NodeType.TRADE.value:
            color = "#2ecc71" if data.get("is_winner", False) else "#e67e22"

        label = data.get("label", nid)
        title_parts = ["<b>" + label + "</b>", "Type: " + ntype]
        for key in ("trade_count", "total_pnl", "wins", "pnl", "pnl_pct", "action", "confidence"):
            if key in data:
                val = data[key]
                if isinstance(val, float):
                    val = round(val, 4)
                title_parts.append(f"{key}: {val}")
        if data.get("trade_count", 0) > 0 and data.get("wins") is not None:
            wr = data["wins"] / data["trade_count"]
            title_parts.append(f"win_rate: {wr:.1%}")

        nodes.append({
            "id": nid,
            "label": label,
            "group": ntype,
            "title": "<br>".join(title_parts),
            "color": color,
            "shape": style["shape"],
            "size": style["size"],
            "raw": {k: v for k, v in data.items()},
        })

    edges: List[Dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        etype = data.get("type", "unknown")
        style = EDGE_STYLE.get(etype, {"color": "#555", "dashes": False, "width": 1, "arrows": "to"})

        edge_label = etype.replace("_", " ")
        if etype == EdgeType.TRANSITIONS_TO.value:
            count = data.get("count", 1)
            edge_label = f"{count}x"

        edges.append({
            "from": u,
            "to": v,
            "label": edge_label,
            "color": {"color": style["color"]},
            "dashes": style["dashes"],
            "width": style.get("width", 1),
            "arrows": style["arrows"],
            "title": etype + ": " + u + " -> " + v,
            "raw": {k: v_ for k, v_ in data.items()},
        })

    return {"nodes": nodes, "edges": edges}


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    gm = _gm_or_error()
    stats = gm.get_stats()
    g = gm._graph
    n = g.number_of_nodes()
    stats["density"] = round(g.number_of_edges() / (n * (n - 1)), 6) if n > 1 else 0
    return stats


@app.get("/api/transitions")
async def get_transitions() -> Dict[str, Any]:
    """Transition probabilities for all regime nodes."""
    gm = _gm_or_error()
    g = gm._graph
    regimes = [
        data.get("label", nid)
        for nid, data in g.nodes(data=True)
        if data.get("type") == NodeType.REGIME.value
    ]
    result: Dict[str, Dict[str, float]] = {}
    for regime in regimes:
        probs = gm.get_regime_transition_probs(regime)
        if probs:
            result[regime] = probs
    return {"transitions": result}


@app.get("/api/strategies")
async def get_strategies() -> Dict[str, Any]:
    """Best strategy per regime."""
    gm = _gm_or_error()
    g = gm._graph
    regimes = [
        data.get("label", nid)
        for nid, data in g.nodes(data=True)
        if data.get("type") == NodeType.REGIME.value
    ]
    result: Dict[str, Any] = {}
    for regime in regimes:
        best = gm.get_best_strategy_for_regime(regime)
        if best:
            result[regime] = best
    return {"strategies": result}


@app.get("/api/correlations/{symbol}")
async def get_correlations(symbol: str) -> Dict[str, Any]:
    gm = _gm_or_error()
    return {"symbol": symbol, "correlations": gm.get_correlated_symbols(symbol)}


@app.get("/api/insight/{regime}/{symbol}")
async def get_insight(regime: str, symbol: str) -> Dict[str, Any]:
    gm = _gm_or_error()
    return gm.get_graph_enhanced_insight(regime, symbol)


# ─── WebSocket Endpoint ───


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── CLI ───


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Graph Memory Dashboard")
    parser.add_argument("--graph-path", default="graph_memory.json", help="Path to graph_memory.json")
    parser.add_argument("--port", type=int, default=8050, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    configure(args.graph_path)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
