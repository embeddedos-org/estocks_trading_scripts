# TradingView Platform Guide

## Overview

TradingView provides Pine Script v5 for building custom strategies, indicators, and scanners. This guide covers setup, usage, and integration with the webhook server for automated order execution.

---

## Pine Script Setup

### Accessing the Pine Editor

1. Open [TradingView](https://www.tradingview.com) and navigate to any chart
2. Click **Pine Editor** at the bottom of the chart
3. Click **Open** → **New indicator** or **New strategy**

### Loading Scripts from This Repository

1. Open the desired `.pine` file from the `tradingview/` directory
2. Copy the entire script content
3. Paste into the Pine Editor
4. Click **Add to Chart** (or **Save** first, then add)

---

## Strategies

### Momentum Breakout (`strategies/momentum_breakout.pine`)

**Logic:** Donchian channel breakout with volume confirmation and ATR trailing stop.

**Parameters:**
| Input | Default | Description |
|-------|---------|-------------|
| `donchianLength` | 20 | Lookback period for Donchian channel |
| `volumeMultiplier` | 1.5 | Volume must exceed SMA(vol) × this |
| `atrLength` | 14 | ATR calculation period |
| `atrMultiplier` | 2.0 | Trailing stop distance in ATR units |
| `positionSizePct` | 10 | Position size as % of equity |

**Usage:**
1. Add to chart as a strategy
2. Open **Strategy Tester** tab to see backtest results
3. Adjust inputs via the gear icon
4. Works best on daily or 4H timeframes for swing trading

### Mean Reversion (`strategies/mean_reversion.pine`)

**Logic:** Bollinger Band mean reversion with RSI filter.

**Entry:** Price touches lower BB (long) or upper BB (short) with RSI confirmation.
**Exit:** Price returns to the middle BB (SMA).

**Best for:** Range-bound markets, mean-reverting instruments.

### Multi-Timeframe (`strategies/multi_timeframe.pine`)

**Logic:** Higher timeframe trend direction (EMA 50/200) filters lower timeframe MACD entries.

**Best for:** Trend-following with precise entries aligned to the larger trend.

---

## Indicators

### Volume Profile (`indicators/volume_profile.pine`)

Displays a session volume profile histogram on the chart showing:
- **POC** (Point of Control) — price level with highest volume
- **VAH** (Value Area High) — upper boundary of 70% volume area
- **VAL** (Value Area Low) — lower boundary of 70% volume area

### Custom RSI (`indicators/custom_rsi.pine`)

Enhanced RSI with:
- Bullish/bearish divergence detection with arrow markers
- Dynamic overbought/oversold levels adjusted by ATR
- Multi-timeframe RSI overlay

### Market Structure (`indicators/market_structure.pine`)

ICT/SMC concepts visualized:
- Swing high/low pivot points
- Break of Structure (BOS) — continuation signal
- Change of Character (ChoCH) — reversal signal
- Order blocks — last opposing candle before a BOS

---

## Scanners

### Gap Scanner (`scanners/gap_scanner.pine`)

Detects and classifies price gaps:
- **Common gaps** — filled within the same session
- **Breakaway gaps** — high volume, trend initiation
- **Exhaustion gaps** — high volume, trend exhaustion

Displays a table with recent gaps, their size, classification, and fill status.

---

## Alert Configuration

### Setting Up Alerts

1. Add a strategy or indicator to your chart
2. Click the **Alert** button (clock icon) on the toolbar
3. Configure:
   - **Condition:** Select the script and alert condition
   - **Actions:** Check **Webhook URL**
   - **Webhook URL:** `http://your-server:5000/webhook`
4. Set the **Alert message** to JSON format:

```json
{
    "symbol": "{{ticker}}",
    "action": "{{strategy.order.action}}",
    "price": {{close}},
    "quantity": {{strategy.order.contracts}},
    "order_type": "market",
    "passphrase": "your-secret-passphrase"
}
```

### TradingView Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{{ticker}}` | Symbol (e.g., "AAPL") |
| `{{close}}` | Current close price |
| `{{strategy.order.action}}` | "buy" or "sell" |
| `{{strategy.order.contracts}}` | Number of contracts/shares |
| `{{time}}` | Alert trigger time |
| `{{exchange}}` | Exchange name |

---

## Webhook Server Deployment

### Local Development

```bash
cd stocks_plugin
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your WEBHOOK_HMAC_SECRET

uvicorn tradingview.webhooks.webhook_server:app --host 0.0.0.0 --port 5000 --reload
```

### Verify Server

```bash
# Health check
curl http://localhost:5000/health

# Test webhook (replace HMAC signature)
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","action":"buy","price":150.0,"quantity":10,"order_type":"market","passphrase":"test"}'
```

### Production Deployment

For TradingView to reach your webhook server, it must be publicly accessible:

1. **Cloud VM** (recommended): Deploy on AWS EC2, DigitalOcean, or similar
2. **Ngrok** (development): `ngrok http 5000` for a temporary public URL
3. **Reverse proxy**: Use nginx/caddy with SSL termination

**Security checklist:**
- ✅ HMAC signature validation enabled
- ✅ Rate limiting configured
- ✅ HTTPS with valid SSL certificate
- ✅ Firewall allows only TradingView IPs + your IP
- ✅ `.env` file not in version control

---

## Broker Routing

The webhook server routes alerts to brokers based on `config.yaml`:

```yaml
broker_routing:
  - pattern: ".*"
    broker: "interactive_brokers"
    account: "default"
```

Supported brokers:
- **Interactive Brokers** — via TWS API (requires TWS/Gateway running)
- **TradeStation** — via REST API (requires API credentials)

### Flow

```
TradingView Alert → Webhook Server → Broker Adapter → Order Placed
                         ↓
                  Alert Dispatcher → Discord/Email/SMS notification
```

---

## Tips & Best Practices

1. **Backtest first** — Always use Strategy Tester before going live
2. **Paper trade** — Run webhook server with paper trading accounts initially
3. **Rate limits** — TradingView Pro+ allows 1 alert per second; adjust rate limits accordingly
4. **Time zones** — Pine Script uses exchange timezone; webhook server uses UTC
5. **Alert persistence** — TradingView alerts can expire; set "Open-ended" for production
6. **Pine Script limits** — Max 500 bars lookback for `request.security()`, max 40 `plot()` calls
