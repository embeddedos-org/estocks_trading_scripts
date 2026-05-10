# System Architecture

## Overview

The stocks_plugin platform provides a unified codebase for developing, testing, and deploying trading automation across four major platforms: TradingView, thinkorswim, Interactive Brokers, and TradeStation.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "Platform-Native Scripts"
        TV[TradingView<br/>Pine Script v5]
        TOS[thinkorswim<br/>thinkScript]
        TS[TradeStation<br/>EasyLanguage]
    end

    subgraph "Python Automation Layer"
        WH[Webhook Server<br/>FastAPI]
        IB[IB Trading Bots<br/>TWS API]
        TSA[TradeStation API<br/>REST Client]
    end

    subgraph "Shared Infrastructure"
        CFG[Config Manager<br/>YAML + .env]
        NOT[Alert Dispatcher<br/>Multi-channel]
        BT[Backtest Engine<br/>OHLCV-based]
    end

    subgraph "External Services"
        TWS[IB TWS / Gateway]
        TSAPI[TradeStation REST API]
        DISC[Discord]
        EMAIL[Email / SMTP]
        SMS[Twilio SMS]
    end

    TV -->|Webhook Alerts| WH
    WH -->|Route Orders| TWS
    WH -->|Route Orders| TSAPI
    WH --> CFG
    WH --> NOT

    IB --> TWS
    IB --> CFG
    IB --> NOT
    IB --> BT

    TSA --> TSAPI
    TSA --> CFG
    TSA --> NOT

    NOT --> DISC
    NOT --> EMAIL
    NOT --> SMS
```

---

## Module Dependency Graph

```mermaid
graph LR
    subgraph shared
        config[shared.config]
        notifier[shared.notifier]
        backtesting[shared.backtesting]
    end

    subgraph tradingview
        pine[Pine Script Files]
        webhook[webhook_server.py]
    end

    subgraph interactive_brokers
        ib_conn[utils.ib_connection]
        ib_order[utils.order_manager]
        ib_hist[data.historical_fetcher]
        ib_rt[data.realtime_stream]
        ib_pairs[strategies.pairs_trading]
        ib_wheel[strategies.options_wheel]
        ib_port[analytics.portfolio_tracker]
        ib_risk[analytics.risk_analyzer]
    end

    subgraph tradestation
        el[EasyLanguage Files]
        ts_order[api.order_router]
        ts_acct[api.account_monitor]
    end

    webhook --> config
    webhook --> notifier

    ib_order --> ib_conn
    ib_order --> notifier
    ib_hist --> ib_conn
    ib_rt --> ib_conn
    ib_pairs --> ib_conn
    ib_pairs --> ib_order
    ib_pairs --> notifier
    ib_wheel --> ib_conn
    ib_wheel --> ib_order
    ib_wheel --> notifier
    ib_port --> ib_conn
    ib_risk --> ib_port

    ts_acct --> ts_order
    ts_acct --> notifier

    backtesting --> config
```

---

## Data Flow

### TradingView Alert → Order Execution

```mermaid
sequenceDiagram
    participant TV as TradingView
    participant WH as Webhook Server
    participant VAL as HMAC Validator
    participant RT as Broker Router
    participant IB as IB TWS
    participant NT as Alert Dispatcher

    TV->>WH: POST /webhook (JSON alert)
    WH->>VAL: Validate HMAC signature
    VAL-->>WH: Valid ✓
    WH->>RT: Route to broker adapter
    RT->>IB: Place order via TWS API
    IB-->>RT: Order confirmation
    RT-->>WH: Execution result
    WH->>NT: Dispatch notification
    NT-->>NT: Discord + Email + Console
```

### IB Pairs Trading Bot Flow

```mermaid
sequenceDiagram
    participant BOT as PairsTradingBot
    participant DATA as HistoricalDataFetcher
    participant TWS as IB TWS/Gateway
    participant OM as OrderManager
    participant NT as AlertDispatcher

    BOT->>DATA: Fetch historical data (Symbol A & B)
    DATA->>TWS: Request bars
    TWS-->>DATA: OHLCV data
    DATA-->>BOT: DataFrames

    BOT->>BOT: Test cointegration (Engle-Granger)
    BOT->>BOT: Calculate hedge ratio (OLS)
    BOT->>BOT: Compute spread z-score

    alt z-score > 2.0 (Short Spread)
        BOT->>OM: Sell Symbol A
        BOT->>OM: Buy Symbol B (hedge ratio qty)
    else z-score < -2.0 (Long Spread)
        BOT->>OM: Buy Symbol A
        BOT->>OM: Sell Symbol B (hedge ratio qty)
    else |z-score| < 0.5 (Exit)
        BOT->>OM: Close all positions
    end

    OM->>TWS: Execute orders
    TWS-->>OM: Fill confirmations
    OM->>NT: Notify on fills
```

---

## Configuration System

```mermaid
graph TD
    ENV[.env file<br/>Secrets & API keys]
    YAML[config.yaml<br/>Structure & parameters]
    ENVVARS[Environment Variables<br/>Runtime overrides]

    ENV --> LOADER[load_config]
    YAML --> LOADER
    ENVVARS --> LOADER

    LOADER --> CONFIG[Merged Config Dict]

    CONFIG --> |brokers| BROKERS[Broker Settings]
    CONFIG --> |notifications| NOTIF[Notification Channels]
    CONFIG --> |strategies| STRAT[Strategy Parameters]
    CONFIG --> |webhook| WH[Webhook Settings]
```

**Priority order** (highest wins):
1. Environment variables
2. `.env` file values
3. `config.yaml` values
4. Default values in code

---

## Notification System

The `AlertDispatcher` supports multi-channel notifications with independent fail-safety:

| Channel | Transport | Use Case |
|---------|-----------|----------|
| Console | stdout | Development, debugging |
| Discord | Webhook POST | Real-time trade alerts |
| Email | SMTP | Daily summaries, critical alerts |
| SMS | Twilio REST API | Critical alerts only |

**Priority Levels:**
- `INFO` — Trade executions, status updates
- `WARNING` — Margin warnings, unusual activity
- `CRITICAL` — System errors, max drawdown breached

Each channel is wrapped in an independent try/except — one channel failing does not block others.

---

## Backtesting Engine

The `BacktestEngine` provides a lightweight, platform-agnostic framework for strategy validation:

```
Input:  OHLCV DataFrame + Strategy Function
Output: BacktestResult (metrics + equity curve + trade log)
```

**Metrics computed:**
- Sharpe Ratio (annualized, √252)
- Sortino Ratio (downside deviation only)
- Maximum Drawdown (peak-to-trough)
- Win Rate (winning / total trades)
- Profit Factor (gross profit / gross loss)
- Total Return (%)
- Equity Curve (list of portfolio values)
- Trade Log (entry/exit price, P&L per trade)

---

## Directory Structure Summary

```
stocks_plugin/
├── shared/              # Foundation — config, notifications, backtesting
├── tradingview/         # Pine Script strategies + webhook server
├── thinkorswim/         # thinkScript studies, scans, watchlists
├── interactive_brokers/ # Python algo trading via TWS API
├── tradestation/        # EasyLanguage + REST API automation
├── docs/                # Architecture & platform guides
└── tests/               # Unit tests for Python modules
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Webhook Server | FastAPI + Uvicorn | Async HTTP server for TradingView alerts |
| IB Connection | ib_async / ibapi | TWS API client libraries |
| TradeStation API | requests + OAuth2 | REST API client |
| Config | PyYAML + python-dotenv | Configuration management |
| Data Analysis | pandas + numpy | OHLCV data processing |
| Statistics | statsmodels + scipy | Cointegration tests, regression |
| Testing | pytest + pytest-asyncio | Unit and integration tests |
| Notifications | requests + smtplib | Multi-channel alerting |
