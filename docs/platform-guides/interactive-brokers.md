# Interactive Brokers Platform Guide

## Overview

Interactive Brokers (IB) provides the most comprehensive API for full algorithmic trading automation. This guide covers TWS/Gateway setup, API configuration, and running the Python trading bots.

---

## TWS / IB Gateway Setup

### Option 1: Trader Workstation (TWS)

TWS is the full trading platform with charting, order entry, and API access.

1. Download TWS from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/tws.php)
2. Install and log in with your IB credentials
3. Enable API access:
   - **Edit** → **Global Configuration** → **API** → **Settings**
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Allow connections from localhost only (for security)
   - Set **Socket port**: `7497` (paper) or `7496` (live)
   - Set **Master API client ID**: leave as 0 or set a specific ID
4. Click **Apply** → **OK**

### Option 2: IB Gateway (Headless)

IB Gateway is lightweight — no GUI, API-only. Recommended for production.

1. Download IB Gateway from the same page
2. Log in with credentials
3. Select **Paper Trading** or **Live Trading**
4. API settings are pre-configured:
   - Paper port: `4002`
   - Live port: `4001`

### Port Reference

| Mode | TWS Port | Gateway Port |
|------|----------|-------------|
| Paper Trading | 7497 | 4002 |
| Live Trading | 7496 | 4001 |

---

## Python Environment Setup

### Installation

```bash
cd stocks_plugin

# Install top-level dependencies
pip install -r requirements.txt

# Install IB-specific dependencies
pip install -r interactive_brokers/requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `ib-async` | High-level async IB API wrapper (recommended) |
| `ibapi` | Official IB Python API (lower-level) |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `statsmodels` | Statistical tests (cointegration) |
| `scipy` | Scientific computing |

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:
```env
IB_HOST=127.0.0.1
IB_PORT=7497          # Use 7497 for paper, 7496 for live
IB_CLIENT_ID=1
IB_ACCOUNT=DU1234567  # Your paper/live account ID
```

---

## Connection Manager

### Using IBConnection

The `IBConnection` factory supports both `ib_async` and `ibapi`:

```python
from interactive_brokers.utils.ib_connection import IBConnection

# Recommended: ib_async backend
conn = IBConnection.create(
    backend="ib_async",
    host="127.0.0.1",
    port=7497,
    client_id=1
)

# As async context manager (auto-connect/disconnect)
async with conn:
    print(f"Connected: {conn.is_connected}")
    # ... trading logic ...

# Or manual connect/disconnect
await conn.connect()
# ... do work ...
conn.disconnect()
```

### Paper vs Live Mode

```python
# Paper trading (default)
conn = IBConnection.create(port=7497)  # TWS paper
conn = IBConnection.create(port=4002)  # Gateway paper

# Live trading
conn = IBConnection.create(port=7496)  # TWS live
conn = IBConnection.create(port=4001)  # Gateway live
```

---

## Running Trading Bots

### Pairs Trading Bot

Statistical arbitrage between two correlated symbols.

```python
from interactive_brokers.utils.ib_connection import IBConnection
from interactive_brokers.strategies.pairs_trading import PairsTradingBot

config = {
    "entry_threshold": 2.0,
    "exit_threshold": 0.5,
    "lookback": 252,
    "zscore_window": 20,
    "position_size": 1000,  # dollars per leg
}

conn = IBConnection.create(port=7497)
bot = PairsTradingBot(
    connection=conn,
    symbol_a="KO",      # Coca-Cola
    symbol_b="PEP",     # PepsiCo
    config=config
)

with conn:
    # Test cointegration first
    is_coint, pvalue = bot.test_cointegration()
    print(f"Cointegrated: {is_coint} (p={pvalue:.4f})")

    if is_coint:
        bot.run()  # Start trading loop
```

### Options Wheel Strategy

The Wheel: sell puts → get assigned → sell calls → repeat.

```python
from interactive_brokers.strategies.options_wheel import OptionsWheelStrategy

config = {
    "put_delta": -0.3,
    "call_delta": 0.3,
    "dte_min": 30,
    "dte_max": 45,
    "position_size": 100,  # shares (1 contract)
}

strategy = OptionsWheelStrategy(
    connection=conn,
    symbol="AAPL",
    config=config
)

with conn:
    strategy.run()
```

---

## Data Collection

### Historical Data

```python
from interactive_brokers.data.historical_fetcher import HistoricalDataFetcher

fetcher = HistoricalDataFetcher(connection=conn)

with conn:
    # Fetch daily bars for 1 year
    df = fetcher.fetch_bars(
        symbol="AAPL",
        duration="1 Y",
        bar_size="1 day",
        what_to_show="TRADES"
    )
    print(df.head())

    # Save to CSV
    fetcher.save_to_csv(df, "data/aapl_daily.csv")

    # Fetch multiple symbols
    data = fetcher.fetch_multiple(
        symbols=["AAPL", "MSFT", "GOOGL"],
        duration="6 M",
        bar_size="1 hour"
    )
```

### Real-Time Streaming

```python
from interactive_brokers.data.realtime_stream import RealtimeDataStream

stream = RealtimeDataStream(connection=conn)

def on_tick(tick_data):
    print(f"{tick_data.symbol}: {tick_data.last} (vol: {tick_data.volume})")

with conn:
    stream.subscribe("AAPL", callback=on_tick)
    stream.subscribe("MSFT", callback=on_tick)

    # Stream runs until interrupted
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stream.unsubscribe("AAPL")
        stream.unsubscribe("MSFT")
```

---

## Portfolio Analytics

```python
from interactive_brokers.analytics.portfolio_tracker import PortfolioTracker
from interactive_brokers.analytics.risk_analyzer import RiskAnalyzer

tracker = PortfolioTracker(connection=conn)
analyzer = RiskAnalyzer(tracker=tracker)

with conn:
    # Portfolio snapshot
    snapshot = tracker.get_snapshot()
    print(f"Equity: ${snapshot.total_equity:,.2f}")
    print(f"P&L: ${snapshot.unrealized_pnl:,.2f}")

    # Sector exposure
    sectors = tracker.sector_exposure()
    for sector, pct in sectors.items():
        print(f"  {sector}: {pct:.1f}%")

    # Risk metrics
    var_95 = analyzer.calculate_var(confidence=0.95)
    sharpe = analyzer.calculate_sharpe()
    max_dd = analyzer.calculate_max_drawdown()
    print(f"VaR (95%): ${var_95:,.2f}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.1%}")
```

---

## Order Management

```python
from interactive_brokers.utils.order_manager import OrderManager

order_mgr = OrderManager(
    connection=conn,
    config={"max_position_pct": 10, "max_daily_loss_pct": 2}
)

with conn:
    # Market order
    order_id = order_mgr.market_order("AAPL", "BUY", 100)

    # Limit order
    order_id = order_mgr.limit_order("MSFT", "BUY", 50, limit_price=350.00)

    # Bracket order (entry + take profit + stop loss)
    order_id = order_mgr.bracket_order(
        "GOOGL", "BUY", 25,
        limit_price=140.00,
        take_profit=155.00,
        stop_loss=135.00
    )

    # Check status
    status = order_mgr.get_order_status(order_id)
    print(f"Order {order_id}: {status.status}")
```

---

## IB API Pacing Rules

IB enforces rate limits on historical data requests:

| Rule | Limit |
|------|-------|
| Identical requests | Wait 15 seconds between identical requests |
| Historical data | Max 60 requests in 10 minutes |
| Market data lines | Max 100 concurrent streams (varies by subscription) |
| Order rate | Max 50 orders per second |

The `HistoricalDataFetcher` automatically handles pacing with built-in delays.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure TWS/Gateway is running and API is enabled |
| Port already in use | Change `client_id` or close other API connections |
| "Not connected" errors | Check firewall, verify host/port settings |
| Market data farm errors | Restart TWS/Gateway, check IB system status |
| Pacing violations | Reduce request frequency, use `_respect_pacing()` |
| Paper trading delays | Paper account data may be 15-20 min delayed |

---

## Tips & Best Practices

1. **Always paper trade first** — Use port 7497/4002 until strategy is validated
2. **Client ID management** — Use unique client IDs for each concurrent connection
3. **Error handling** — IB API can disconnect unexpectedly; use auto-reconnect
4. **Market hours** — Some data requests only work during market hours
5. **Account permissions** — Enable "Market Data" and "Trading" API permissions in Account Management
6. **Logging** — Enable verbose logging during development: `import logging; logging.basicConfig(level=logging.DEBUG)`
