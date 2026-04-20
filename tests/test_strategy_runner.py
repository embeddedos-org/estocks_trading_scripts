"""
Tests for strategies/runner.py
================================

Covers:
- _parse_params() with various formats
- register_strategy() and STRATEGY_REGISTRY retrieval
- cmd_backtest() for standard, factor, ML strategies
- --chart flag works for factor strategy (verify fix)
- cmd_list() output
- _generate_synthetic_data / _generate_synthetic_universe
- _load_data with synthetic and file inputs
- _print_result / _result_to_dict
"""

import sys
import os
import argparse
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pandas as pd
import pytest

# sys.path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.runner import (
    _parse_params,
    _generate_synthetic_data,
    _generate_synthetic_universe,
    _load_data,
    _print_result,
    _result_to_dict,
    cmd_list,
    cmd_backtest,
    _render_chart,
)
from strategies import STRATEGY_REGISTRY, register_strategy, list_strategies
from shared.backtesting.backtest_engine_v2 import BacktestResultV2


# ─── _parse_params tests ───


class TestParseParams:
    """Tests for _parse_params() parameter parsing."""

    def test_empty_string_returns_empty_dict(self):
        assert _parse_params("") == {}

    def test_none_returns_empty_dict(self):
        assert _parse_params(None) == {}

    def test_single_int_param(self):
        result = _parse_params("rsi_length=14")
        assert result == {"rsi_length": 14}
        assert isinstance(result["rsi_length"], int)

    def test_single_float_param(self):
        result = _parse_params("stop_loss=0.03")
        assert result == {"stop_loss": 0.03}
        assert isinstance(result["stop_loss"], float)

    def test_single_string_param(self):
        result = _parse_params("strategy=trend_following")
        assert result == {"strategy": "trend_following"}
        assert isinstance(result["strategy"], str)

    def test_bool_true_param(self):
        result = _parse_params("use_filter=true")
        assert result == {"use_filter": True}
        assert isinstance(result["use_filter"], bool)

    def test_bool_false_param(self):
        result = _parse_params("trailing_stop=false")
        assert result == {"trailing_stop": False}
        assert isinstance(result["trailing_stop"], bool)

    def test_bool_case_insensitive(self):
        assert _parse_params("a=True") == {"a": True}
        assert _parse_params("a=FALSE") == {"a": False}

    def test_multiple_params(self):
        result = _parse_params("rsi_oversold=25,rsi_overbought=75,bb_std=2.5")
        assert result == {
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "bb_std": 2.5,
        }

    def test_params_with_spaces(self):
        result = _parse_params("  key1 = 10 , key2 = 3.14  ")
        assert result == {"key1": 10, "key2": 3.14}

    def test_pair_without_equals_skipped(self):
        result = _parse_params("valid=1,invalid_no_equals,another=2")
        assert result == {"valid": 1, "another": 2}

    def test_value_with_equals_sign(self):
        result = _parse_params("formula=a=b")
        assert result == {"formula": "a=b"}

    def test_negative_int(self):
        result = _parse_params("offset=-5")
        assert result == {"offset": -5}
        assert isinstance(result["offset"], int)

    def test_negative_float(self):
        result = _parse_params("threshold=-0.15")
        assert result == {"threshold": -0.15}
        assert isinstance(result["threshold"], float)


# ─── Registry tests ───


class TestStrategyRegistry:
    """Tests for register_strategy() and STRATEGY_REGISTRY retrieval."""

    def test_known_strategies_registered(self):
        expected = {"trend_following", "mean_reversion", "breakout", "factor", "ml", "rl", "self_learning"}
        for name in expected:
            assert name in STRATEGY_REGISTRY, f"{name} not registered"

    def test_register_custom_strategy(self):
        @register_strategy("test_custom")
        class TestCustomStrategy:
            """A custom test strategy."""
            pass

        assert "test_custom" in STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY["test_custom"] is TestCustomStrategy
        # cleanup
        del STRATEGY_REGISTRY["test_custom"]

    def test_list_strategies_returns_descriptions(self):
        strategies = list_strategies()
        assert isinstance(strategies, dict)
        assert len(strategies) > 0
        for name, desc in strategies.items():
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_strategy_from_params(self):
        cls = STRATEGY_REGISTRY["trend_following"]
        strategy = cls.from_params({"fast_ma_length": 5, "slow_ma_length": 15})
        assert strategy.config.fast_ma_length == 5
        assert strategy.config.slow_ma_length == 15

    def test_strategy_from_params_ignores_unknown(self):
        cls = STRATEGY_REGISTRY["mean_reversion"]
        strategy = cls.from_params({"rsi_length": 10, "unknown_param": 999})
        assert strategy.config.rsi_length == 10


# ─── Synthetic data generation ───


class TestSyntheticData:
    """Tests for synthetic data generators."""

    def test_generate_synthetic_data_shape(self):
        df = _generate_synthetic_data(n_bars=100, seed=1)
        assert len(df) == 100
        assert set(df.columns) >= {"date", "open", "high", "low", "close", "volume"}

    def test_generate_synthetic_data_deterministic(self):
        df1 = _generate_synthetic_data(n_bars=50, seed=42)
        df2 = _generate_synthetic_data(n_bars=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_synthetic_data_positive_prices(self):
        df = _generate_synthetic_data(n_bars=500, seed=42)
        assert (df["close"] > 0).all()
        assert (df["high"] >= df["low"]).all()

    def test_generate_synthetic_universe_shape(self):
        df = _generate_synthetic_universe(n_stocks=5, n_bars=100, seed=1)
        assert df.shape == (100, 5)
        assert all(col.startswith("STOCK_") for col in df.columns)

    def test_generate_synthetic_universe_deterministic(self):
        df1 = _generate_synthetic_universe(n_stocks=3, n_bars=50, seed=42)
        df2 = _generate_synthetic_universe(n_stocks=3, n_bars=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


# ─── _load_data tests ───


class TestLoadData:
    """Tests for _load_data."""

    def test_load_synthetic_keyword(self):
        df = _load_data("synthetic")
        assert len(df) == 500
        assert "close" in df.columns

    def test_load_synthetic_uppercase(self):
        df = _load_data("SYNTHETIC")
        assert len(df) == 500

    def test_load_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,open,high,low,close,volume\n")
            f.write("2020-01-01,100,105,95,102,1000000\n")
            f.write("2020-01-02,102,106,98,104,1100000\n")
            tmp_path = f.name
        try:
            df = _load_data(tmp_path)
            assert len(df) == 2
            assert "close" in df.columns
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file_exits(self):
        with pytest.raises(SystemExit):
            _load_data("nonexistent_file_xyz.csv")


# ─── _print_result / _result_to_dict tests ───


class TestResultHelpers:
    """Tests for result display and serialization."""

    def _make_result(self, **kwargs):
        defaults = dict(
            total_return=0.15, cagr=0.12, sharpe_ratio=1.5,
            sortino_ratio=2.0, max_drawdown=-0.10, calmar_ratio=1.2,
            win_rate=0.55, profit_factor=1.8, total_trades=50,
            long_trades=30, short_trades=20, avg_win=500.0,
            avg_loss=-300.0, expectancy=120.0, avg_trade_duration=5.2,
            max_consecutive_wins=8, max_consecutive_losses=4,
            alpha=0.0, beta=0.0,
        )
        defaults.update(kwargs)
        return BacktestResultV2(**defaults)

    def test_print_result_no_crash(self, capsys):
        result = self._make_result()
        _print_result(result)
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "15.00%" in captured.out

    def test_print_result_with_alpha_beta(self, capsys):
        result = self._make_result(alpha=0.05, beta=1.1)
        _print_result(result)
        captured = capsys.readouterr()
        assert "Alpha" in captured.out
        assert "Beta" in captured.out

    def test_result_to_dict_keys(self):
        result = self._make_result()
        d = _result_to_dict(result)
        expected_keys = {
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
            "total_trades", "long_trades", "short_trades", "avg_win",
            "avg_loss", "expectancy", "avg_trade_duration",
            "max_consecutive_wins", "max_consecutive_losses",
        }
        assert set(d.keys()) == expected_keys

    def test_result_to_dict_values_match(self):
        result = self._make_result(total_return=0.25, sharpe_ratio=2.0)
        d = _result_to_dict(result)
        assert d["total_return"] == 0.25
        assert d["sharpe_ratio"] == 2.0


# ─── cmd_backtest tests ───


class TestCmdBacktest:
    """Tests for cmd_backtest() with mocked engine."""

    def _make_args(self, **kwargs):
        defaults = dict(
            strategy="trend_following",
            data="synthetic",
            params="",
            capital=100_000,
            chart=False,
            output="",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("strategies.runner.BacktestEngineV2")
    def test_backtest_standard_strategy(self, mock_engine_cls):
        mock_result = BacktestResultV2(total_return=0.1, total_trades=10)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        args = self._make_args(strategy="trend_following")
        cmd_backtest(args)

        mock_engine.load_data.assert_called_once()
        mock_engine.run.assert_called_once()

    def test_backtest_unknown_strategy_exits(self):
        args = self._make_args(strategy="nonexistent_strategy_xyz")
        with pytest.raises(SystemExit):
            cmd_backtest(args)

    @patch("strategies.runner._render_chart")
    @patch("strategies.runner.BacktestEngineV2")
    def test_backtest_factor_with_chart_flag(self, mock_engine_cls, mock_render):
        """Verify fix: --chart flag works for factor strategy."""
        mock_result = BacktestResultV2(total_return=0.05, total_trades=5)

        factor_cls = STRATEGY_REGISTRY["factor"]
        with patch.object(factor_cls, "run_backtest", return_value=mock_result):
            args = self._make_args(strategy="factor", chart=True)
            cmd_backtest(args)

        mock_render.assert_called_once_with(mock_result)

    @patch("strategies.runner.BacktestEngineV2")
    def test_backtest_with_params(self, mock_engine_cls):
        mock_result = BacktestResultV2(total_return=0.2)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        args = self._make_args(
            strategy="mean_reversion",
            params="rsi_oversold=25,rsi_overbought=75",
        )
        cmd_backtest(args)
        mock_engine.run.assert_called_once()

    @patch("strategies.runner.BacktestEngineV2")
    def test_backtest_output_to_file(self, mock_engine_cls):
        mock_result = BacktestResultV2(total_return=0.1, sharpe_ratio=1.5)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            args = self._make_args(strategy="trend_following", output=tmp_path)
            cmd_backtest(args)

            with open(tmp_path) as f:
                data = json.load(f)
            assert data["total_return"] == 0.1
            assert data["sharpe_ratio"] == 1.5
        finally:
            os.unlink(tmp_path)

    @patch("strategies.runner.BacktestEngineV2")
    def test_backtest_ml_strategy_calls_train(self, mock_engine_cls):
        mock_result = BacktestResultV2(total_return=0.05)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        ml_cls = STRATEGY_REGISTRY["ml"]
        with patch.object(ml_cls, "train") as mock_train:
            args = self._make_args(strategy="ml")
            cmd_backtest(args)
            mock_train.assert_called_once()

    @patch("strategies.runner._render_chart")
    @patch("strategies.runner.BacktestEngineV2")
    def test_backtest_standard_with_chart(self, mock_engine_cls, mock_render):
        mock_result = BacktestResultV2(total_return=0.1)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        args = self._make_args(strategy="breakout", chart=True)
        cmd_backtest(args)
        mock_render.assert_called_once_with(mock_result)


# ─── cmd_list tests ───


class TestCmdList:
    """Tests for cmd_list()."""

    def test_cmd_list_prints_strategies(self, capsys):
        args = argparse.Namespace()
        cmd_list(args)
        captured = capsys.readouterr()
        assert "Available Strategies" in captured.out
        assert "trend_following" in captured.out


# ─── _render_chart tests ───


class TestRenderChart:
    """Tests for _render_chart with missing deps."""

    def test_render_chart_import_error(self, capsys):
        with patch.dict("sys.modules", {"shared.visualization.chart_renderer": None}):
            result = BacktestResultV2()
            _render_chart(result)
            captured = capsys.readouterr()
            assert "Warning" in captured.out or "unavailable" in captured.out or captured.out == ""
