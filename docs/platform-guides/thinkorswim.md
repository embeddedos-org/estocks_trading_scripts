# thinkorswim Platform Guide

## Overview

thinkorswim (by Charles Schwab) provides thinkScript for building custom studies, scans, watchlist columns, and conditional order strategies. This guide covers importing scripts, configuring scans, and setting up automation.

---

## Importing thinkScript Studies

### Method 1: Direct Paste

1. Open **thinkorswim** desktop platform
2. Navigate to **Charts** tab
3. Click **Studies** → **Edit Studies**
4. Click **Create** (or **New**)
5. Clear the default code
6. Copy/paste the `.ts` file content from this repository
7. Click **OK** to save

### Method 2: Shared Link

1. In thinkorswim, go to **Setup** → **Open Shared Item**
2. Paste the shared study link (if available)
3. The study is added to your study list

### Adding to Charts

1. Go to **Charts** → **Studies** → **Edit Studies**
2. Find your custom study in the left panel under **Personal**
3. Double-click or click **Add** to apply it to the chart
4. Adjust inputs in the **Customize** section

---

## Studies

### Custom MACD (`studies/custom_macd.ts`)

An enhanced MACD with 4-color histogram and divergence detection.

**Inputs:**
| Input | Default | Description |
|-------|---------|-------------|
| `fastLength` | 12 | Fast EMA period |
| `slowLength` | 26 | Slow EMA period |
| `signalLength` | 9 | Signal line period |
| `divergenceLookback` | 5 | Bars to look back for divergence |

**Features:**
- 4-color histogram: dark/light green (positive), dark/light red (negative)
- Signal line crossover arrows and alerts
- Zero-line cross detection
- Bullish/bearish divergence markers

**Chart Setup:**
- Appears as a **lower study**
- Recommended: pair with price chart and volume study

### Relative Strength (`studies/relative_strength.ts`)

Mansfield Relative Strength vs SPX benchmark.

**Inputs:**
| Input | Default | Description |
|-------|---------|-------------|
| `referenceSymbol` | "SPX" | Benchmark symbol |
| `maLength` | 52 | Moving average period for RS line |

**Interpretation:**
- RS above MA → stock outperforming benchmark (green background)
- RS below MA → stock underperforming (red background)
- Rising RS + above zero → strong relative strength
- Use for sector rotation and stock selection

---

## Stock Hacker Scans

### Setting Up Custom Scans

1. Go to **Scan** tab → **Stock Hacker**
2. Click **Add Study Filter**
3. Select **Custom** → choose your scan script
4. Configure filter conditions (e.g., `is greater than`, `is true`)
5. Set universe: S&P 500, All Stocks, custom watchlist
6. Click **Scan**

### Unusual Volume (`scans/unusual_volume.ts`)

Finds stocks with unusual volume activity.

**Scan Criteria:**
- Current volume > 2× the 20-day average
- Price movement confirmation (|close - open| > 0.5%)
- Direction indicator: up-volume or down-volume

**Recommended Setup:**
1. Add as Study Filter in Stock Hacker
2. Filter: `volume_ratio` is greater than `2.0`
3. Universe: S&P 500 or All Optionable
4. Sort by volume_ratio descending

**Use Cases:**
- Identify institutional activity
- Find potential breakout candidates
- Pre-earnings volume spikes

---

## Watchlist Columns

### Adding Custom Columns

1. Go to **MarketWatch** → **Quotes** tab
2. Right-click any column header → **Customize**
3. Click **Custom Quotes** at the bottom
4. Select your thinkScript watchlist script
5. The column appears in your watchlist

### Sector Rotation (`watchlists/sector_rotation.ts`)

Momentum-based scoring for sector ETFs.

**Scoring Formula:**
- 1-week ROC × 0.3 + 1-month ROC × 0.4 + 3-month ROC × 0.3

**Color Coding:**
| Score Range | Color | Interpretation |
|-------------|-------|---------------|
| > 5 | Dark Green | Strong momentum |
| 2 to 5 | Light Green | Positive momentum |
| -2 to 2 | Yellow | Neutral |
| -5 to -2 | Orange | Negative momentum |
| < -5 | Red | Strong negative momentum |

**Recommended Watchlist:**
XLF, XLK, XLE, XLV, XLI, XLY, XLP, XLU, XLC, XLRE, XLB

---

## Strategies & Conditional Orders

### Earnings Play (`strategies/earnings_play.ts`)

Automated earnings setup strategy.

**Logic:**
1. Detect upcoming earnings using `HasEarnings()`
2. Check if IV Rank exceeds threshold
3. Buy N days before earnings
4. Sell 1 day before the announcement

**Inputs:**
| Input | Default | Description |
|-------|---------|-------------|
| `daysBefore` | 5 | Days before earnings to enter |
| `ivRankThreshold` | 50 | Minimum IV Rank to trigger entry |

### Setting Up Conditional Orders

1. Right-click a stock → **Buy/Sell Custom** → **Conditional Order**
2. Under **Condition**, select a custom study as the trigger
3. Set the condition (e.g., study value crosses above 0)
4. Configure the order type and duration
5. Click **Confirm and Send**

**Note:** Conditional orders require the thinkorswim platform to be running.

---

## Schwab API Integration

The Schwab API (successor to TD Ameritrade API) enables programmatic access:

### Authentication

1. Register at [Schwab Developer Portal](https://developer.schwab.com)
2. Create an application to get `app_key` and `app_secret`
3. Follow the OAuth2 authorization flow
4. Store tokens in `.env` file

### Capabilities

- **Account data:** Balances, positions, order history
- **Market data:** Quotes, price history, option chains
- **Order placement:** Market, limit, stop, complex orders
- **Streaming:** Real-time quotes via WebSocket

---

## Tips & Best Practices

1. **ThinkBack** — Use for historical replay and strategy validation
2. **OnDemand** — thinkorswim's replay mode for paper testing with historical data
3. **Workspaces** — Save chart/study configurations as workspaces for quick switching
4. **Performance** — Limit concurrent studies to avoid platform lag (max 5-6 per chart)
5. **Alerts** — Set study-based alerts for notification without keeping the platform open
6. **Scanner limits** — Stock Hacker scans can run on at most ~3 study filters efficiently
