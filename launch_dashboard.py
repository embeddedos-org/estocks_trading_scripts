#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch Graph Memory Dashboard
===============================

Entry point to start the Graphify Memory Dashboard web server.

Usage:
    python launch_dashboard.py [--graph-path graph_memory.json] [--port 8050]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.dashboard.app import app, configure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Graphify Memory Dashboard"
    )
    parser.add_argument(
        "--graph-path",
        default="graph_memory.json",
        help="Path to graph_memory.json (default: graph_memory.json)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Server port (default: 8050)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}")
        print("Run 'python demo_graph_memory.py' first to generate sample data.")
        sys.exit(1)

    configure(str(graph_path))

    print(f"Starting Graphify Memory Dashboard...")
    print(f"  Graph: {graph_path}")
    print(f"  URL:   http://localhost:{args.port}")
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
