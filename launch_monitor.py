"""
Launch: Monitor Mode (Watch Only — No Trades)
================================================

The agent analyzes all symbols, computes decisions, logs everything,
but does NOT execute any trades. Safe for observation and learning.

All decisions logged to:
  ~/.stocks_plugin/logs/decisions.jsonl
  ~/.stocks_plugin/data/trade_memory.db
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from shared.daemon.live_runner import LiveRunner

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

if __name__ == "__main__":
    print("=" * 60)
    print("  MONITOR MODE — WATCH ONLY")
    print("  Analyzing markets + news, logging decisions, NO trades")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("=" * 60)

    runner = LiveRunner(
        symbols=SYMBOLS,
        mode="monitor",
        interval_seconds=300,
        use_news=True,
        models=["regime"],
    )
    runner.start(train_first=True)
