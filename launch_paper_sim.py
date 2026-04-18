"""
Launch: Paper Simulation (No Broker Needed)
=============================================

Runs the AI agent with Yahoo Finance data and simulated paper trading.
No IB Gateway or brokerage account required.

Monitors: AAPL, MSFT, GOOGL, AMZN, META, SPY, QQQ, NVDA, TSLA
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from shared.daemon.live_runner import LiveRunner

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

if __name__ == "__main__":
    print("=" * 60)
    print("  PAPER SIMULATION MODE")
    print("  No broker connection — using Yahoo Finance + simulated trades")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("=" * 60)

    runner = LiveRunner(
        symbols=SYMBOLS,
        mode="paper",
        interval_seconds=300,  # 5 minutes
        use_news=True,
        models=["regime"],  # Fast training — regime classifier only
    )
    runner.start(train_first=True)
