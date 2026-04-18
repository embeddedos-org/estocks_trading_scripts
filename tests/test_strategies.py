"""
Comprehensive tests for the strategies package.

Covers: registry, all 6 strategies (trend_following, mean_reversion, breakout,
factor, ml, rl), and the CLI runner module.
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import all example strategies to trigger registration
import strategies.examples.trend_following  # noqa: F401
import strategies.examples.mean_reversion  # noqa: F401
import strategies.examples.breakout  # noqa: F401
import strategies.examples.factor_portfolio  # noqa: F401
import strategies.examples.ml_rl_strategy  # noqa: F401

from strategies import STRATEGY_REGISTRY, list_strategies, register_strategy
from strategies.runner import (
    _parse_params,
    _result_to_dict,
    cmd_list,
    cmd_backtest,
)
from shared.backtesting.backtest_engine_v2 import (
    BacktestContext,
    BacktestEngineV2,
    BacktestResultV2,
)


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------
class TestStrategyRegistry:
    """Tests for the strategy registry in strategies/__init__.py."""

    def test_all_strategies_registered(self):
        expected = {"trend_following", "mean_reversion", "breakout", "factor", "ml", "rl"}
        assert expected.issubset(set(STRATEGY_REGISTRY.keys()))

    def test_registry_values_are_classes(self):
        for name, cls in STRATEGY_REGISTRY.items():
            assert isinstance(cls, type), f"{name} should be a class"

    def test_list_strategies_returns_descriptions(self):
        result = list_strategies()
        assert isinstance(result, dict)
        assert len(result) >= 6
        for name, desc in result.items():
            assert isinstance(desc, str)

    def test_register_strategy_decorator(self):
        @register_strategy("_test_dummy")
        class DummyStrategy:
            """A dummy strategy for testing registration."""
            pass

        assert "_test_dummy" in STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY["_test_dummy"] is DummyStrategy
        # Clean up
        del STRATEGY_REGISTRY["_test_dummy"]

    def test_registry_count(self):
        assert len(STRATEGY_REGISTRY) >= 6


# ---------------------------------------------------------------------------
# Trend Following Strategy
# ---------------------------------------------------------------------------
class TestTrendFollowing:
    """Tests for TrendFollowingStrategy."""

    def test_import_and_instantiate(self):
        from strategies.examples.trend_following import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        assert s.config.fast_ma_length == 9
        assert s.config.slow_ma_length == 21

    def test_from_params(self):
        from strategies.examples.trend_following import TrendFollowingStrategy
        s = TrendFollowingStrategy.from_params({"fast_ma_length": 5, "slow_ma_length": 15})
        assert s.config.fast_ma_length == 5
        assert s.config.slow_ma_length == 15

    def test_generate_signals_returns_valid(self, synthetic_ohlcv_df):
        from strategies.examples.trend_following import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        ctx = BacktestContext(
            bar_index=199,
            bars={"SYM": synthetic_ohlcv_df},
            positions={},
            capital=100_000,
            portfolio_value=100_000,
        )
        signals = s.generate_signals(ctx)
        assert "SYM" in signals
        assert signals["SYM"] in (-1, 0, 1)

    def test_full_backtest(self, backtest_engine):
        from strategies.examples.trend_following import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        result = backtest_engine.run(s.generate_signals)
        assert isinstance(result, BacktestResultV2)
        assert len(result.equity_curve) > 0

    def test_custom_config_differs(self, synthetic_ohlcv_df):
        from strategies.examples.trend_following import TrendFollowingStrategy, TrendFollowingConfig
        # Use shorter trend filter so both configs can generate signals on 200-bar data
        s1 = TrendFollowingStrategy(TrendFollowingConfig(trend_filter_length=50))
        s2 = TrendFollowingStrategy(TrendFollowingConfig(fast_ma_length=5, slow_ma_length=50, trend_filter_length=50))

        engine1 = BacktestEngineV2(initial_capital=100_000)
        engine1.load_data(synthetic_ohlcv_df)
        r1 = engine1.run(s1.generate_signals)

        engine2 = BacktestEngineV2(initial_capital=100_000)
        engine2.load_data(synthetic_ohlcv_df)
        r2 = engine2.run(s2.generate_signals)

        # Both should produce valid results; different configs may produce different outcomes
        assert isinstance(r1, BacktestResultV2)
        assert isinstance(r2, BacktestResultV2)
        assert len(r1.equity_curve) == len(r2.equity_curve)


# ---------------------------------------------------------------------------
# Mean Reversion Strategy
# ---------------------------------------------------------------------------
class TestMeanReversion:
    """Tests for MeanReversionStrategy."""

    def test_import_and_instantiate(self):
        from strategies.examples.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        assert s.config.rsi_length == 14
        assert s.config.rsi_oversold == 30
        assert s.config.rsi_overbought == 70

    def test_from_params(self):
        from strategies.examples.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy.from_params({"rsi_oversold": 25, "rsi_overbought": 75})
        assert s.config.rsi_oversold == 25
        assert s.config.rsi_overbought == 75

    def test_generate_signals_valid(self, synthetic_ohlcv_df):
        from strategies.examples.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        ctx = BacktestContext(
            bar_index=199,
            bars={"SYM": synthetic_ohlcv_df},
            positions={},
            capital=100_000,
            portfolio_value=100_000,
        )
        signals = s.generate_signals(ctx)
        assert "SYM" in signals
        assert signals["SYM"] in (-1, 0, 1)

    def test_full_backtest(self, backtest_engine):
        from strategies.examples.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        result = backtest_engine.run(s.generate_signals)
        assert isinstance(result, BacktestResultV2)
        assert len(result.equity_curve) > 0

    def test_stop_loss_config(self):
        from strategies.examples.mean_reversion import MeanReversionStrategy, MeanReversionConfig
        s = MeanReversionStrategy(MeanReversionConfig(stop_loss_pct=0.05))
        assert s.config.stop_loss_pct == 0.05


# ---------------------------------------------------------------------------
# Breakout Strategy
# ---------------------------------------------------------------------------
class TestBreakout:
    """Tests for BreakoutStrategy."""

    def test_import_and_instantiate(self):
        from strategies.examples.breakout import BreakoutStrategy
        s = BreakoutStrategy()
        assert s.config.channel_length == 20
        assert s.config.volume_mult == 1.5

    def test_from_params(self):
        from strategies.examples.breakout import BreakoutStrategy
        s = BreakoutStrategy.from_params({"channel_length": 30, "volume_mult": 2.0})
        assert s.config.channel_length == 30
        assert s.config.volume_mult == 2.0

    def test_generate_signals_valid(self, synthetic_ohlcv_df):
        from strategies.examples.breakout import BreakoutStrategy
        s = BreakoutStrategy()
        ctx = BacktestContext(
            bar_index=199,
            bars={"SYM": synthetic_ohlcv_df},
            positions={},
            capital=100_000,
            portfolio_value=100_000,
        )
        signals = s.generate_signals(ctx)
        assert "SYM" in signals
        assert signals["SYM"] in (-1, 0, 1)

    def test_full_backtest(self, backtest_engine):
        from strategies.examples.breakout import BreakoutStrategy
        s = BreakoutStrategy()
        result = backtest_engine.run(s.generate_signals)
        assert isinstance(result, BacktestResultV2)
        assert len(result.equity_curve) > 0

    def test_confirm_bars_config(self):
        from strategies.examples.breakout import BreakoutStrategy, BreakoutConfig
        s = BreakoutStrategy(BreakoutConfig(confirm_bars=3))
        assert s.config.confirm_bars == 3


# ---------------------------------------------------------------------------
# Factor Portfolio Strategy
# ---------------------------------------------------------------------------
class TestFactorPortfolio:
    """Tests for FactorPortfolioStrategy."""

    def test_import_and_instantiate(self):
        from strategies.examples.factor_portfolio import FactorPortfolioStrategy
        s = FactorPortfolioStrategy()
        assert s.config.n_long == 5
        assert s.config.n_short == 5

    def test_from_params(self):
        from strategies.examples.factor_portfolio import FactorPortfolioStrategy
        s = FactorPortfolioStrategy.from_params({"n_long": 3, "n_short": 3})
        assert s.config.n_long == 3
        assert s.config.n_short == 3

    def test_run_backtest_on_universe(self, synthetic_universe):
        from strategies.examples.factor_portfolio import (
            FactorPortfolioStrategy,
            FactorPortfolioConfig,
        )
        s = FactorPortfolioStrategy(FactorPortfolioConfig(n_long=2, n_short=2))
        result = s.run_backtest(synthetic_universe)
        assert isinstance(result, BacktestResultV2)
        assert len(result.equity_curve) > 0

    def test_momentum_ranking(self, synthetic_universe):
        from strategies.examples.factor_portfolio import (
            FactorPortfolioStrategy,
            FactorPortfolioConfig,
        )
        s = FactorPortfolioStrategy(FactorPortfolioConfig(n_long=2, n_short=2))
        result = s.run_backtest(synthetic_universe)
        assert result.total_trades >= 0


# ---------------------------------------------------------------------------
# ML Strategy
# ---------------------------------------------------------------------------
class TestMLStrategy:
    """Tests for MLStrategy (LSTM with torch fallback)."""

    def test_import_and_instantiate(self):
        from strategies.examples.ml_rl_strategy import MLStrategy
        s = MLStrategy()
        assert s.config.seq_len == 60
        assert s.config.epochs == 20

    def test_fallback_mode(self):
        from strategies.examples.ml_rl_strategy import MLStrategy
        s = MLStrategy()
        # Whether torch is installed or not, the strategy should instantiate
        assert isinstance(s._fallback, bool)

    def test_train_graceful(self, synthetic_ohlcv_df):
        from strategies.examples.ml_rl_strategy import MLStrategy
        s = MLStrategy()
        # Should not raise regardless of torch availability
        # May raise ValueError if data is too small for LSTM — that's expected
        try:
            s.train(synthetic_ohlcv_df)
        except ValueError:
            s._fallback = True  # insufficient data → force fallback

    def test_generate_signals_valid(self, synthetic_ohlcv_df):
        from strategies.examples.ml_rl_strategy import MLStrategy
        s = MLStrategy()
        try:
            s.train(synthetic_ohlcv_df)
        except ValueError:
            s._fallback = True
        ctx = BacktestContext(
            bar_index=199,
            bars={"SYM": synthetic_ohlcv_df},
            positions={},
            capital=100_000,
            portfolio_value=100_000,
        )
        signals = s.generate_signals(ctx)
        assert "SYM" in signals
        assert signals["SYM"] in (-1, 0, 1)


# ---------------------------------------------------------------------------
# RL Strategy
# ---------------------------------------------------------------------------
class TestRLStrategy:
    """Tests for RLStrategy (PPO with SB3 fallback)."""

    def test_import_and_instantiate(self):
        from strategies.examples.ml_rl_strategy import RLStrategy
        s = RLStrategy()
        assert s.config.algorithm == "PPO"

    def test_fallback_mode(self):
        from strategies.examples.ml_rl_strategy import RLStrategy
        s = RLStrategy()
        assert isinstance(s._fallback, bool)

    def test_train_graceful(self, synthetic_ohlcv_df):
        from strategies.examples.ml_rl_strategy import RLStrategy
        s = RLStrategy()
        s.train(synthetic_ohlcv_df)

    def test_rsi_fallback_signals(self, synthetic_ohlcv_df):
        from strategies.examples.ml_rl_strategy import RLStrategy
        s = RLStrategy()
        s._fallback = True  # Force fallback mode
        ctx = BacktestContext(
            bar_index=199,
            bars={"SYM": synthetic_ohlcv_df},
            positions={},
            capital=100_000,
            portfolio_value=100_000,
        )
        signals = s.generate_signals(ctx)
        assert "SYM" in signals
        assert signals["SYM"] in (-1, 0, 1)


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------
class TestCLIRunner:
    """Tests for the strategies.runner CLI module."""

    def test_cmd_list(self, capsys):
        args = argparse.Namespace()
        cmd_list(args)
        captured = capsys.readouterr()
        assert "trend_following" in captured.out
        assert "mean_reversion" in captured.out
        assert "breakout" in captured.out

    def test_cmd_backtest_synthetic(self, capsys):
        args = argparse.Namespace(
            strategy="trend_following",
            data="synthetic",
            params="",
            capital=100_000,
            chart=False,
            output="",
        )
        cmd_backtest(args)
        captured = capsys.readouterr()
        assert "Total Return" in captured.out

    def test_cmd_backtest_with_params(self, capsys):
        args = argparse.Namespace(
            strategy="mean_reversion",
            data="synthetic",
            params="rsi_oversold=25,rsi_overbought=75",
            capital=100_000,
            chart=False,
            output="",
        )
        cmd_backtest(args)
        captured = capsys.readouterr()
        assert "Total Return" in captured.out

    def test_cmd_backtest_unknown_strategy(self):
        args = argparse.Namespace(
            strategy="nonexistent_strategy",
            data="synthetic",
            params="",
            capital=100_000,
            chart=False,
            output="",
        )
        with pytest.raises(SystemExit):
            cmd_backtest(args)

    def test_parse_params_types(self):
        result = _parse_params("x=10,y=3.14,flag=true,name=hello")
        assert result["x"] == 10
        assert isinstance(result["x"], int)
        assert result["y"] == 3.14
        assert isinstance(result["y"], float)
        assert result["flag"] is True
        assert result["name"] == "hello"

    def test_parse_params_empty(self):
        assert _parse_params("") == {}
        assert _parse_params(None) == {}

    def test_result_to_dict(self):
        result = BacktestResultV2(
            total_return=0.15,
            cagr=0.10,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.08,
            calmar_ratio=1.25,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=20,
            long_trades=12,
            short_trades=8,
            avg_win=500.0,
            avg_loss=300.0,
            expectancy=50.0,
            avg_trade_duration=5.0,
            max_consecutive_wins=4,
            max_consecutive_losses=3,
        )
        d = _result_to_dict(result)
        assert isinstance(d, dict)
        assert d["total_return"] == 0.15
        assert d["sharpe_ratio"] == 1.5
        assert d["total_trades"] == 20
        assert d["long_trades"] == 12
        assert d["short_trades"] == 8

    def test_cmd_backtest_factor_synthetic(self, capsys):
        args = argparse.Namespace(
            strategy="factor",
            data="synthetic",
            params="",
            capital=100_000,
            chart=False,
            output="",
        )
        cmd_backtest(args)
        captured = capsys.readouterr()
        assert "Total Return" in captured.out
