"""
Launch: IB Paper Trading (Requires IB Gateway/TWS running)
============================================================

Connects to Interactive Brokers paper trading account via IB Gateway.
The agent makes real decisions and places real paper orders.

Prerequisites:
  1. IB Gateway/TWS running on port 7497 (paper)
  2. API enabled in IB Gateway settings
  3. See docs/ib_setup_guide.txt for setup instructions
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from shared.daemon.live_runner import LiveRunner

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "7497"))  # 7497=paper, 7496=live

if __name__ == "__main__":
    print("=" * 60)
    print("  IB PAPER TRADING MODE")
    print(f"  Connecting to IB Gateway at {IB_HOST}:{IB_PORT}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("  Orders will be placed on your IB PAPER account")
    print("=" * 60)

    runner = LiveRunner(
        symbols=SYMBOLS,
        mode="paper",
        interval_seconds=300,
        use_news=True,
        models=["regime"],
        broker="ib",
        broker_config={"host": IB_HOST, "port": IB_PORT, "client_id": 1},
    )
    runner.start(train_first=True)
