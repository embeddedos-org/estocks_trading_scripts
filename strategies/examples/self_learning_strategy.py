"""
Self-Learning Strategy — AI Agent that Improves from History
================================================================

Wraps the SelfLearningAgent as a registered strategy compatible with
the BacktestEngineV2 and strategy runner CLI.

The self-learning strategy:
1. Trains all ML models (Regime, LSTM, Transformer, RL) on provided data
2. Walks through each bar making autonomous decisions
3. Records every decision + outcome to TradeMemory (SQLite)
4. Adapts ensemble weights based on which models performed best
5. Detects model degradation and recommends retraining

This is the most advanced strategy in stocks_plugin — it combines
every model and learns from its own trading history.

Usage:
    # Via runner CLI:
    python -m strategies.runner backtest --strategy self_learning --data synthetic

    # Programmatically:
    from strategies.examples.self_learning_strategy import SelfLearningStrategy
    strategy = SelfLearningStrategy()
    strategy.train(df)
    engine = BacktestEngineV2()
    engine.load_data(df)
    result = engine.run(strategy.generate_signals)
"""

from __future__ import annotations

import sys
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.backtesting.backtest_engine_v2 import (
    BacktestContext,
    BacktestEngineV2,
    BacktestResultV2,
)
from strategies import register_strategy


@dataclass
class SelfLearningConfig:
    """Configuration for the SelfLearningStrategy."""

    # Which models to train (comma-separated or list)
    models: str = "regime"  # "regime,lstm,transformer,rl" for full suite
    buy_threshold: float = 0.15
    sell_threshold: float = -0.15
    min_confidence: float = 0.3
    adaptive_thresholds: bool = True
    stop_loss_pct: float = 0.05  # 5% stop loss
    use_enricher: bool = True


@register_strategy("self_learning")
class SelfLearningStrategy:
    """AI agent strategy that learns from its own trading history.

    Trains ML models on historical data and combines their predictions
    via an adaptive ensemble. Records every trade with full context
    to a SQLite database, enabling the agent to learn what works
    in different market regimes.

    Models used:
    - Regime Classifier (LightGBM): identifies TRENDING/RANGING/VOLATILE
    - LSTM Predictor (PyTorch): forecasts next-day returns
    - Transformer Predictor (PyTorch): attention-based price prediction
    - RL Agent (PPO/SB3): reinforcement learning trader
    - Momentum: simple fallback (always available)
    """

    def __init__(self, config: Optional[SelfLearningConfig] = None) -> None:
        self.config = config or SelfLearningConfig()
        self._agent = None
        self._trained = False
        self._fallback = False
        self._entry_prices: Dict[str, float] = {}
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception:
                pass

        # Create agent
        try:
            from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

            db_path = os.path.join(tempfile.gettempdir(), "sl_backtest_memory.db")
            agent_config = AgentConfig(
                db_path=db_path,
                buy_threshold=self.config.buy_threshold,
                sell_threshold=self.config.sell_threshold,
                min_confidence=self.config.min_confidence,
                adaptive_thresholds=self.config.adaptive_thresholds,
            )
            self._agent = SelfLearningAgent(agent_config)
        except Exception as e:
            print(f"  [SelfLearning] Agent init failed ({e}), using momentum fallback")
            self._fallback = True

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "SelfLearningStrategy":
        config = SelfLearningConfig(**{
            k: v for k, v in params.items() if hasattr(SelfLearningConfig, k)
        })
        return cls(config)

    def train(self, df: pd.DataFrame) -> None:
        """Train all ML models on historical data."""
        if self._fallback or self._agent is None:
            print("  [SelfLearning] Using momentum fallback (ML dependencies unavailable)")
            return

        models = [m.strip() for m in self.config.models.split(",")]
        print(f"  [SelfLearning] Training models: {models}")

        try:
            results = self._agent.train(df, models=models, verbose=True)
            self._trained = True
            print(f"  [SelfLearning] Training complete: {list(results.keys())}")
        except Exception as e:
            print(f"  [SelfLearning] Training failed: {e}. Using momentum fallback.")
            self._fallback = True

    def generate_signals(self, ctx: BacktestContext) -> Dict[str, int]:
        """Generate signals using the self-learning agent."""
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < 60:
                signals[sym] = 0
                continue

            current_price = float(df["close"].iloc[-1])
            current_pos = ctx.positions.get(sym, 0)

            # Stop loss check on held positions
            if current_pos != 0 and sym in self._entry_prices:
                entry = self._entry_prices[sym]
                if current_pos > 0 and current_price < entry * (1 - self.config.stop_loss_pct):
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    continue
                elif current_pos < 0 and current_price > entry * (1 + self.config.stop_loss_pct):
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    continue

            if self._fallback or self._agent is None:
                # Enricher gate
                enricher_ok = True
                if getattr(self, "_enricher", None) and current_pos == 0:
                    enriched = self._enricher.enrich(sym, df)
                    blocked, _ = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False

                raw_signal = self._momentum_signal(df)
                if not enricher_ok and raw_signal != 0 and current_pos == 0:
                    signals[sym] = 0
                else:
                    signals[sym] = raw_signal
                if signals[sym] == 1 and current_pos <= 0:
                    self._entry_prices[sym] = current_price
                elif signals[sym] == -1 and current_pos >= 0:
                    self._entry_prices[sym] = current_price
                continue

            try:
                # Enricher gate for new entries
                enricher_ok = True
                if getattr(self, "_enricher", None) and current_pos == 0:
                    enriched = self._enricher.enrich(sym, df)
                    blocked, _ = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False

                decision = self._agent.decide(df, symbol=sym)
                action = decision.get("action", "HOLD")

                if action == "BUY" and enricher_ok:
                    signals[sym] = 1
                    self._entry_prices[sym] = current_price
                elif action == "SELL":
                    signals[sym] = -1
                    self._entry_prices[sym] = current_price
                else:
                    signals[sym] = 0

                # Record outcome from previous trade if position changed
                prev_pos = ctx.positions.get(sym, 0)

                if prev_pos != 0 and signals[sym] != prev_pos:
                    entry = self._entry_prices.get(sym, current_price)
                    pnl = (current_price - entry) * prev_pos
                    self._entry_prices.pop(sym, None)
                    self._agent.record_outcome(
                        exit_price=current_price,
                        pnl=pnl,
                        holding_period_bars=1,
                    )

            except Exception as e:
                signals[sym] = self._momentum_signal(df)

        return signals

    def _momentum_signal(self, df: pd.DataFrame) -> int:
        """Simple momentum fallback when ML models aren't available."""
        close = df["close"]
        if len(close) < 20:
            return 0

        mom_20 = float(close.iloc[-1] / close.iloc[-20] - 1)
        mom_5 = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
        score = 0.6 * mom_20 + 0.4 * mom_5

        if score > 0.01:
            return 1
        elif score < -0.01:
            return -1
        return 0


def _generate_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    price = 100.0
    prices = []
    for i in range(n_bars):
        regime = np.sin(2 * np.pi * i / 120)
        drift = 0.0003 * regime
        ret = drift + rng.randn() * 0.015
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.005)
        low = price * (1 - abs(rng.randn()) * 0.005)
        prices.append({
            "date": dates[i],
            "open": price * (1 + rng.randn() * 0.002),
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(prices)


def run_example() -> BacktestResultV2:
    """Run the self-learning strategy on synthetic data."""
    print("=" * 60)
    print("SELF-LEARNING AI AGENT — STRATEGY EXAMPLE")
    print("=" * 60)

    df = _generate_data()

    strategy = SelfLearningStrategy(SelfLearningConfig(
        models="regime",  # Use just regime for quick demo
        min_confidence=0.2,
    ))
    strategy.train(df)

    engine = BacktestEngineV2(initial_capital=100_000)
    engine.load_data(df)
    result = engine.run(strategy.generate_signals)

    print(f"\n  Total Return:  {result.total_return:>10.2%}")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:>10.4f}")
    print(f"  Max Drawdown:  {result.max_drawdown:>10.2%}")
    print(f"  Total Trades:  {result.total_trades:>10d}")
    print(f"  Win Rate:      {result.win_rate:>10.2%}")

    # Show agent's learned insights
    if strategy._agent:
        print("\n--- Agent Self-Assessment ---")
        perf = strategy._agent.get_performance(lookback_days=365)
        print(f"  Total trades in memory: {perf.get('total_trades', 0)}")
        weights = strategy._agent.get_weight_summary()
        for model, w in weights.items():
            print(f"  {model}: effective_weight={w.get('effective_weight', 0):.3f}")

    print("\n" + "=" * 60)
    return result


if __name__ == "__main__":
    run_example()
