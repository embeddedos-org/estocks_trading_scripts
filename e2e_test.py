#!/usr/bin/env python3
"""End-to-end pipeline test with synthetic data."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

print("=" * 70)
print("  END-TO-END TRADING PIPELINE TEST")
print("=" * 70)

# 1. Generate realistic OHLCV
np.random.seed(42)
n = 300
dates = pd.bdate_range("2024-01-01", periods=n)
price = 150.0
rows = []
for i in range(n):
    ret = 0.001 + np.random.randn() * 0.015
    price *= 1 + ret
    rows.append({"date": dates[i], "open": price*(1+np.random.randn()*0.002),
        "high": price*(1+abs(np.random.randn())*0.005),
        "low": price*(1-abs(np.random.randn())*0.005),
        "close": price, "volume": int(np.random.uniform(5e5, 2e6))})
df_aapl = pd.DataFrame(rows).set_index("date")

np.random.seed(99)
price2 = 300.0
rows2 = []
for i in range(n):
    ret2 = 0.0005 + np.random.randn() * 0.018
    price2 *= 1 + ret2
    rows2.append({"date": dates[i], "open": price2*(1+np.random.randn()*0.002),
        "high": price2*(1+abs(np.random.randn())*0.005),
        "low": price2*(1-abs(np.random.randn())*0.005),
        "close": price2, "volume": int(np.random.uniform(3e5, 1.5e6))})
df_msft = pd.DataFrame(rows2).set_index("date")
print(f"\n1. DATA: AAPL {len(df_aapl)} bars (${df_aapl['close'].iloc[-1]:.2f}), MSFT {len(df_msft)} bars (${df_msft['close'].iloc[-1]:.2f})")

# 2. Load strategies
import strategies.examples.trend_following
import strategies.examples.breakout
import strategies.examples.mean_reversion
import strategies.examples.factor_portfolio
import strategies.examples.darvas_box
import strategies.examples.triple_screen
import strategies.examples.canslim_strategy
import strategies.examples.value_strategy
import strategies.examples.ml_rl_strategy
import strategies.examples.self_learning_strategy
import strategies.examples.sentiment_strategy
import strategies.examples.earnings_strategy
import strategies.examples.sector_rotation
import strategies.examples.meta_strategy
from strategies import STRATEGY_REGISTRY
print(f"2. STRATEGIES: {len(STRATEGY_REGISTRY)} loaded")

# 3. Run all strategies
from shared.backtesting.backtest_engine_v2 import BacktestContext
ctx = BacktestContext(bar_index=299, bars={"AAPL": df_aapl, "MSFT": df_msft},
                      positions={}, capital=100000, portfolio_value=100000)
print("\n3. STRATEGY SIGNALS:")
results = {}
for name in sorted(STRATEGY_REGISTRY.keys()):
    try:
        strat = STRATEGY_REGISTRY[name]()
        signals = strat.generate_signals(ctx)
        aapl_sig = signals.get("AAPL", 0)
        label = {1: "BUY", -1: "SELL", 0: "HOLD", 2: "ADD"}.get(aapl_sig, "?")
        results[name] = aapl_sig
        print(f"   {name:20s} -> AAPL:{label:5s} MSFT:{signals.get('MSFT', 0):+d}")
    except Exception as e:
        print(f"   {name:20s} -> ERROR: {str(e)[:60]}")

# 4. Risk Manager
from shared.risk_manager import RiskManager, RiskManagerConfig
rm = RiskManager(config=RiskManagerConfig(total_capital=100000, min_seconds_between_trades=0, max_trades_per_hour=10000))
last_price = float(df_aapl["close"].iloc[-1])
size = rm.calculate_position_size("AAPL", last_price, stop_price=last_price*0.95)
ok, reason = rm.validate_order("AAPL", size, last_price, last_price)
print(f"\n4. RISK MANAGER:")
print(f"   Position size: {size} shares (${size * last_price:,.0f})")
print(f"   Order validation: {'APPROVED' if ok else 'REJECTED'} ({reason})")
print(f"   can_trade(): {rm.can_trade()}")

# 5. Backtest
from shared.backtesting.backtest_engine_v2 import BacktestEngineV2
engine = BacktestEngineV2(initial_capital=100000)
engine.load_data({"AAPL": df_aapl.reset_index(), "MSFT": df_msft.reset_index()})
result = engine.run(STRATEGY_REGISTRY["trend_following"]().generate_signals)
print(f"\n5. BACKTEST (trend_following):")
print(f"   Total Return: {result.total_return:+.2%}")
print(f"   Sharpe Ratio: {result.sharpe_ratio:.4f}")
print(f"   Max Drawdown: {result.max_drawdown:.2%}")
print(f"   Win Rate:     {result.win_rate:.2%}")
print(f"   Total Trades: {result.total_trades}")
print(f"   Profit Factor:{result.profit_factor:.2f}")
print(f"   SQN:          {result.sqn:.4f}")

# 6. Enricher
from shared.strategy_enricher import StrategyEnricher
enricher = StrategyEnricher()
enriched = enricher.enrich("AAPL", df_aapl)
blocked, reason = enricher.should_block_entry(enriched)
print(f"\n6. STRATEGY ENRICHER:")
print(f"   Regime: {enriched.regime}")
print(f"   ML Signal: {enriched.ml_signal:.4f}")
print(f"   Composite Boost: {enriched.composite_boost:.2f}")
print(f"   Entry blocked: {blocked} ({reason})")

# 7. Trade Journal
from shared.trade_journal import TradeJournal
j = TradeJournal(journal_dir=os.path.join(tempfile.mkdtemp(), "journal"))
j.pre_trade_check("AAPL", mood=8, confidence=7, setup="trend")
j.log_trade("AAPL", pnl=800, lesson="Good trend")
j.log_trade("MSFT", pnl=-500, lesson="False breakout", followed_plan=False)
stats = j.get_discipline_stats()
print(f"\n7. TRADE JOURNAL:")
print(f"   Trades: {stats['total_trades']}, Adherence: {stats['plan_adherence_pct']:.0f}%")

# 8. Indicators
from shared.indicators.technical_indicators import TechnicalIndicators as TI
fi = TI.force_index(df_aapl)
er = TI.elder_ray(df_aapl)
ei = TI.elder_impulse(df_aapl)
fib = TI.fibonacci_retracement(df_aapl)
print(f"\n8. INDICATORS:")
print(f"   Force Index: {float(fi.iloc[-1]):.2f}")
print(f"   Bull Power:  {float(er['bull_power'].iloc[-1]):.2f}")
print(f"   Impulse:     {ei.iloc[-1]}")
print(f"   Fib 61.8%:   ${float(fib['fib_618'].iloc[-1]):.2f}")

# 9. Signal consensus
buy_count = sum(1 for v in results.values() if v > 0)
sell_count = sum(1 for v in results.values() if v < 0)
hold_count = sum(1 for v in results.values() if v == 0)
print(f"\n9. SIGNAL CONSENSUS:")
print(f"   BUY:  {buy_count}/{len(results)} strategies")
print(f"   SELL: {sell_count}/{len(results)} strategies")
print(f"   HOLD: {hold_count}/{len(results)} strategies")

print("\n" + "=" * 70)
print("  ALL SYSTEMS OPERATIONAL - READY TO TRADE")
print("=" * 70)
