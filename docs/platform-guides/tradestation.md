# TradeStation Platform Guide

## Overview

TradeStation provides EasyLanguage for building strategies and indicators natively, plus a REST API for external Python automation. This guide covers importing EasyLanguage scripts, configuring RadarScreen, and setting up the API.

---

## EasyLanguage Setup

### Importing Strategies & Indicators

1. Open **TradeStation** desktop platform
2. Go to **File** → **New** → **Analysis Technique**
3. Select type: **Strategy**, **Indicator**, or **Function**
4. Name your script (e.g., "Trend_Following")
5. Clear the default code
6. Copy/paste the `.el` file content from this repository
7. Click **Verify** (F3) to compile
8. Click **File** → **Close** to save

### Applying to Charts

**Strategies:**
1. Open a chart
2. Click **Insert** → **Strategy**
3. Select your custom strategy
4. Configure inputs in the dialog
5. Click **OK** — strategy signals appear on chart

**Indicators:**
1. Open a chart
2. Click **Insert** → **Indicator**
3. Select your custom indicator
4. Configure inputs and display settings
5. Click **OK**

---

## Strategies

### Trend Following (`strategies/trend_following.el`)

Dual moving average crossover with ADX filter and Chandelier Exit.

**Inputs:**
| Input | Default | Description |
|-------|---------|-------------|
| `FastMALength` | 20 | Fast moving average period |
| `SlowMALength` | 50 | Slow moving average period |
| `ADXThreshold` | 25 | Minimum ADX for trend strength |
| `ATRPeriod` | 22 | ATR period for Chandelier Exit |
| `ATRMultiplier` | 3.0 | ATR multiplier for stop distance |
| `RiskPercent` | 2.0 | Risk per trade as % of equity |

**Logic:**
- **Long entry:** Fast MA crosses above Slow MA AND ADX > 25
- **Short entry:** Fast MA crosses below Slow MA AND ADX > 25
- **Exit:** Chandelier Exit (ATR trailing stop)
- **Position sizing:** Fixed fractional based on risk %

**Best used on:** Daily or 60-minute charts for swing trading.

---

## Indicators

### Adaptive Moving Average (`indicators/adaptive_moving_avg.el`)

Kaufman Adaptive Moving Average (KAMA) — adapts speed based on market efficiency.

**Inputs:**
| Input | Default | Description |
|-------|---------|-------------|
| `Length` | 10 | Efficiency ratio lookback |
| `FastLength` | 2 | Fast smoothing period |
| `SlowLength` | 30 | Slow smoothing period |

**Interpretation:**
- **Green line** — KAMA is rising (bullish trend)
- **Red line** — KAMA is falling (bearish trend)
- **Flat line** — Market is choppy (KAMA adapts to low efficiency)
- Faster than SMA in trends, slower in chop

---

## RadarScreen Configuration

RadarScreen is TradeStation's real-time multi-symbol monitoring tool.

### Setting Up RadarScreen

1. Open **Apps** → **RadarScreen**
2. Add symbols to the left panel (type symbol + Enter)
3. Right-click a column header → **Insert Analysis Technique**
4. Select your custom indicator
5. The indicator values update in real-time

### Sector Momentum Scanner (`scanners/sector_momentum.el`)

**Setup:**
1. Add sector ETFs to RadarScreen: XLF, XLK, XLE, XLV, XLI, XLY, XLP, XLU, XLC, XLRE, XLB
2. Insert the `Sector_Momentum` indicator as a column
3. Sort by score descending to see strongest sectors

**Color Coding:**
| Score Range | Color | Meaning |
|-------------|-------|---------|
| > 10 | Dark Green | Very strong momentum |
| 5 to 10 | Green | Positive momentum |
| -5 to 5 | Yellow | Neutral |
| -10 to -5 | Orange | Weak momentum |
| < -10 | Red | Very weak momentum |

**Trading Application:**
- Go long stocks in sectors with high scores
- Avoid or short stocks in sectors with low scores
- Rebalance weekly based on momentum changes

---

## TradeStation API Setup

### Step 1: Register for API Access

1. Log in to [TradeStation Client Center](https://clientcenter.tradestation.com)
2. Navigate to **API Access** or **Developer Portal**
3. Create a new application
4. Note your **Client ID** and **Client Secret**
5. Set **Redirect URI** (e.g., `http://localhost:8080/callback`)

### Step 2: Configure Credentials

```bash
cp .env.example .env
```

Edit `.env`:
```env
TS_CLIENT_ID=your_client_id_here
TS_CLIENT_SECRET=your_client_secret_here
TS_REDIRECT_URI=http://localhost:8080/callback
TS_REFRESH_TOKEN=your_refresh_token_here
```

### Step 3: OAuth2 Authentication Flow

The TradeStation API uses OAuth2 Authorization Code flow:

1. **Authorization URL:** Direct user to TradeStation login
2. **Authorization Code:** User grants access, receives code via redirect
3. **Token Exchange:** Exchange code for access + refresh tokens
4. **Token Refresh:** Use refresh token to get new access tokens (auto-handled by `TradeStationOrderRouter`)

```python
from tradestation.api.order_router import TradeStationOrderRouter

config = {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "redirect_uri": "http://localhost:8080/callback",
    "refresh_token": "your_refresh_token"
}

router = TradeStationOrderRouter(config=config)

# Place a market order
order_id = router.place_market_order(
    account_id="123456789",
    symbol="AAPL",
    action="BUY",
    quantity=100
)

# Place a bracket order
order_id = router.place_bracket_order(
    account_id="123456789",
    symbol="MSFT",
    action="BUY",
    quantity=50,
    limit_price=350.00,
    profit_target=365.00,
    stop_loss=340.00
)
```

### Step 4: Account Monitoring

```python
from tradestation.api.account_monitor import AccountMonitor

monitor = AccountMonitor(
    order_router=router,
    config={
        "margin_warning_pct": 80,
        "max_drawdown_pct": 5,
        "position_concentration_pct": 25
    }
)

# One-time check
balances = monitor.get_balances("123456789")
positions = monitor.get_positions("123456789")

# Continuous monitoring (background thread)
monitor.start_monitoring("123456789", interval_seconds=60)

# Generate daily summary
summary = monitor.generate_daily_summary("123456789")
print(summary)

# Stop monitoring
monitor.stop_monitoring()
```

---

## API Endpoints Reference

Base URL: `https://api.tradestation.com/v3`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/brokerage/accounts` | GET | List accounts |
| `/brokerage/accounts/{id}/balances` | GET | Account balances |
| `/brokerage/accounts/{id}/positions` | GET | Open positions |
| `/brokerage/accounts/{id}/orders` | GET | Order history |
| `/orderexecution/orders` | POST | Place order |
| `/orderexecution/orders/{id}` | DELETE | Cancel order |
| `/marketdata/quotes/{symbols}` | GET | Real-time quotes |
| `/marketdata/barcharts/{symbol}` | GET | Historical bars |
| `/marketdata/options/chains/{symbol}` | GET | Option chains |

---

## EasyLanguage vs Python API

| Feature | EasyLanguage | Python API |
|---------|-------------|-----------|
| Backtesting | Built-in Strategy Tester | Custom (use BacktestEngine) |
| Real-time signals | Native chart integration | Via REST polling |
| Order execution | Automated in platform | Via REST API |
| Walk-Forward Optimization | Built-in | Manual |
| Custom analytics | Limited | Full Python ecosystem |
| Multi-broker | TradeStation only | Any broker with API |
| RadarScreen | Native | Not available |

**Recommendation:** Use EasyLanguage for strategy development, backtesting, and RadarScreen monitoring. Use Python API for custom analytics, multi-account management, and cross-broker automation.

---

## Tips & Best Practices

1. **Verify before applying** — Always compile (F3) EasyLanguage code before adding to charts
2. **Walk-Forward Optimizer** — Use for robust parameter selection, avoid overfitting
3. **Strategy properties** — Set max bars back, commission, slippage in strategy properties
4. **RadarScreen limits** — Max ~100 symbols with complex indicators; keep it simple for speed
5. **API rate limits** — TradeStation API allows 120 requests/minute; the `order_router.py` handles this
6. **Token management** — Refresh tokens expire after 90 days of inactivity; re-authenticate if needed
7. **Simulation vs Live** — Test with simulation account before enabling live trading
8. **Data subscriptions** — Real-time data requires active TradeStation subscription
