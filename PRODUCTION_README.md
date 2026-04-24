# Stocks Plugin — Automated Trading System

Production-ready algorithmic trading framework with 15 strategies, 7 data sources,
7-layer risk management, and comprehensive backtesting.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a backtest
python -m strategies.examples.trend_following

# Run all tests
python -m pytest tests/ -v --tb=short -m "not slow and not ml"
```

## Architecture

```
stocks_plugin/
├── shared/                          # Core infrastructure
│   ├── risk_manager.py              # 7-layer risk engine (thread-safe)
│   ├── risk_manager_unified.py      # Cross-strategy portfolio gate
│   ├── strategy_enricher.py         # Multi-source data enrichment
│   ├── trade_journal.py             # Human psychology/discipline journal
│   ├── data/
│   │   └── public_data_fetcher.py   # OHLCV, news, fundamentals, earnings
│   ├── indicators/
│   │   ├── technical_indicators.py  # 35+ indicators (TA-Lib → pandas-ta → manual)
│   │   ├── candlestick_patterns.py  # 14 patterns + cup-and-handle
│   │   └── multi_timeframe.py       # Higher-timeframe trend confirmation
│   ├── backtesting/
│   │   └── backtest_engine_v2.py    # Multi-asset backtester with R-multiples/SQN
│   └── ml/
│       ├── news_sentiment.py        # FinBERT → VADER → keyword sentiment
│       ├── ensemble_predictor.py    # 6-model weighted ensemble
│       ├── regime_classifier.py     # LightGBM TRENDING/RANGING/VOLATILE
│       ├── trade_memory.py          # SQLite trade journal (ML)
│       └── self_learning_agent.py   # Adaptive weight optimization
├── strategies/
│   ├── __init__.py                  # Strategy registry
│   └── examples/                    # 15 registered strategies
└── tests/                           # 288+ tests
```

## 15 Trading Strategies

| # | Strategy | Key | Based On | Data Sources |
|---|----------|-----|----------|-------------|
| 1 | Trend Following | `trend_following` | EMA crossover + ADX | Price, Volume, Fund, News, Earnings, ML, Regime |
| 2 | Breakout | `breakout` | Donchian channel breakout | All 7 |
| 3 | Mean Reversion | `mean_reversion` | RSI + Bollinger Bands | All 7 |
| 4 | Factor Portfolio | `factor` | 12-1 month momentum L/S | All 7 |
| 5 | Darvas Box | `darvas_box` | Box ceiling breakout | All 7 |
| 6 | Triple Screen | `triple_screen` | Elder 3-screen system | All 7 |
| 7 | CAN SLIM | `canslim` | O'Neil 7-criteria scoring | All 7 |
| 8 | Graham Value | `value` | Fundamental value scoring | All 7 |
| 9 | ML (LSTM) | `ml` | Deep learning prediction | All 7 |
| 10 | RL (PPO) | `rl` | Reinforcement learning | All 7 |
| 11 | Self-Learning | `self_learning` | Adaptive ML ensemble | All 7 |
| 12 | News Sentiment | `sentiment` | Headline sentiment | All 7 |
| 13 | Earnings Calendar | `earnings` | Earnings surprise drift | All 7 |
| 14 | Sector Rotation | `sector_rotation` | Sector ETF momentum | All 7 |
| 15 | Meta Ensemble | `meta_ensemble` | 5-component weighted composite | All 7 |

### Data Sources (7)

| Source | Provider | Method | Cache TTL |
|--------|----------|--------|-----------|
| 📈 Price (OHLCV) | Yahoo Finance | `fetch_ohlcv()` | 5 min |
| 📊 Volume | Yahoo Finance | Included in OHLCV | 5 min |
| 💰 Fundamentals | Yahoo Finance | `fetch_fundamentals()` | 1 hour |
| 📰 News | YF + Google RSS | `fetch_news_headlines()` | 5 min |
| 📅 Earnings | Yahoo Finance | `fetch_earnings_dates()` | 5 min |
| 🧠 ML Signal | Momentum proxy | `StrategyEnricher._enrich_ml()` | Per-bar |
| 🌊 Regime | ADX + ATR | `StrategyEnricher._enrich_regime()` | Per-bar |

## 7-Layer Risk Management

```
Layer 7 ─ PORTFOLIO HEAT ──── Max 20% of equity at risk at any time
Layer 6 ─ POSITION CAP ────── Max 25% equity / 10K shares per position
Layer 5 ─ CIRCUIT BREAKER ─── 10% drawdown → 24h pause
Layer 4 ─ MONTHLY CAP ─────── Elder 6% monthly loss limit (opt-in)
Layer 3 ─ DAILY LIMIT ─────── $5,000/day hard stop
Layer 2 ─ COOLDOWN ─────────── 30-min pause after 3 consecutive losses
Layer 1 ─ PER-TRADE RISK ──── 2% of equity per trade
```

### Production Safety Controls

| Control | Default | Config Field |
|---------|---------|-------------|
| Max position size (% equity) | 25% | `max_position_pct_equity` |
| Max shares per order | 10,000 | `max_shares_per_order` |
| Price deviation check | ±10% | `max_price_deviation_pct` |
| Max short positions | 5 | `max_short_positions` |
| Max short exposure | 30% | `max_short_exposure_pct` |
| Min avg daily volume | 50,000 | `min_avg_volume` |
| Max position % of ADV | 5% | `max_position_pct_adv` |
| Market hours enforcement | Off | `enforce_market_hours` |
| State persistence | SQLite WAL | `persist_path` |

### Pre-Order Validation

```python
from shared.risk_manager import RiskManager, RiskManagerConfig

rm = RiskManager(config=RiskManagerConfig(total_capital=100000))

# Every order must pass validate_order() before execution
ok, reason = rm.validate_order(
    symbol="AAPL",
    shares=100,
    price=195.0,
    last_price=194.5,
    direction="LONG",
    avg_daily_volume=50_000_000,
)
if ok:
    # Place order
    pass
else:
    print(f"Order rejected: {reason}")
```

## Thread Safety

`RiskManager` is thread-safe for live trading:
- `record_trade()`, `add_position()`, `remove_position()` acquire `_state_lock`
- `_save_state()` acquires `_persist_lock` for SQLite writes
- Safe for concurrent broker fill callbacks

## Strategy Enricher

All 15 strategies use `StrategyEnricher` to gate entries with multi-source checks:

```python
from shared.strategy_enricher import StrategyEnricher

enricher = StrategyEnricher()
enriched = enricher.enrich("AAPL", df)

# Check before every entry:
blocked, reason = enricher.should_block_entry(enriched)
if blocked:
    # Skip this trade — bad sentiment, poor fundamentals, or earnings blackout
    pass
```

**Entry is blocked when:**
- News sentiment < -0.4 (strongly bearish headlines)
- Fundamental quality score < 0.4 (poor P/E, debt, margins)
- Earnings announcement within 2 trading days (gap risk)

## Configuration for Live Trading

```python
config = RiskManagerConfig(
    total_capital=100_000.0,
    risk_per_trade_pct=1.0,           # Conservative 1%
    max_daily_loss=2_000.0,           # 2% daily max
    max_monthly_loss=4_000.0,         # 4% monthly max
    max_drawdown_pct=8.0,             # Circuit breaker at 8%
    max_trades_per_hour=5,
    max_consecutive_losses=3,
    cooldown_seconds=3600,            # 1 hour cooldown
    max_position_pct_equity=15.0,     # 15% max per position
    max_shares_per_order=5_000,
    max_price_deviation_pct=5.0,
    enforce_market_hours=True,
    min_avg_volume=100_000,
    persist_path="~/.stocks_plugin/risk_state.db",
    enable_pyramiding=False,
)
```

## Testing

```bash
# Core tests (fast, no ML dependencies)
python -m pytest tests/test_new_features.py tests/test_production_safety.py \
    tests/test_shared_risk_manager.py tests/test_shared_indicators.py \
    tests/test_strategy_examples.py -v

# Full suite
python -m pytest tests/ -v --tb=short -m "not slow and not ml"

# Production safety only
python -m pytest tests/test_production_safety.py -v
```

**Current status: 288 passed, 12 skipped, 0 failures.**

## Trading Books Implemented

| Book | Author | Coverage | Key Features |
|------|--------|----------|-------------|
| Trading in the Zone | Mark Douglas | ✅ | Pre-trade checklist, discipline journal |
| The Disciplined Trader | Mark Douglas | ✅ | Cooldown, trade frequency limits |
| Daily Trading Coach | Steenbarger | ✅ | Trade journal, performance by mood |
| Market Wizards | Schwager | ✅ | Risk per trade, daily limits, circuit breakers |
| Reminiscences | Livermore | ✅ | Pyramiding, trend following, stops |
| Trade Your Way | Van Tharp | ✅ | Fixed fractional, Kelly, R-multiples, SQN |
| Trading for a Living | Elder | ✅ | Force Index, Elder-ray, Impulse, Triple Screen, 6% rule |
| Darvas Box Method | Darvas | ✅ | Box breakout strategy |
| TA of Financial Markets | Murphy | ✅ | 35+ indicators, Fibonacci, candlesticks |
| How to Make Money | O'Neil | ✅ | CAN SLIM, cup-and-handle |
| Intelligent Investor | Graham | ✅ | Value strategy, fundamental scoring |

## Disclaimer

This software is for educational and research purposes. Past backtest performance
does not guarantee future results. Always paper trade for 30+ days before using
real money. The authors are not responsible for any trading losses.
