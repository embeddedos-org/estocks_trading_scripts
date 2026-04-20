"""Tests for BacktestEngine — runs a SMA crossover strategy on synthetic data."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from shared.backtesting.backtest_engine import BacktestEngine, BacktestResult


def _generate_synthetic_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a trending + mean-reverting pattern."""
    rng = np.random.RandomState(seed)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_bars)]

    trend = np.linspace(100, 130, n_bars)
    noise = rng.normal(0, 1.5, n_bars).cumsum()
    close = trend + noise
    close = np.maximum(close, 10.0)

    high = close + rng.uniform(0.5, 2.0, n_bars)
    low = close - rng.uniform(0.5, 2.0, n_bars)
    open_ = close + rng.normal(0, 0.5, n_bars)
    volume = rng.randint(100_000, 1_000_000, n_bars)

    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _sma_crossover_strategy(index: int, row: dict, position: int, capital: float) -> int:
    """Simple SMA crossover: buy when short SMA > long SMA, sell otherwise.

    This is a simplified version — the engine passes individual rows,
    so we use price as a proxy and track state via position.
    """
    if index < 1:
        return 0

    if not hasattr(_sma_crossover_strategy, "_prices"):
        _sma_crossover_strategy._prices = []

    _sma_crossover_strategy._prices.append(row["close"])
    prices = _sma_crossover_strategy._prices

    short_window = 10
    long_window = 30

    if len(prices) < long_window:
        return 0

    short_sma = np.mean(prices[-short_window:])
    long_sma = np.mean(prices[-long_window:])

    if short_sma > long_sma and position == 0:
        return 1
    elif short_sma < long_sma and position == 1:
        return -1
    return 0


@pytest.fixture(autouse=True)
def reset_strategy_state():
    """Reset strategy state between tests."""
    if hasattr(_sma_crossover_strategy, "_prices"):
        del _sma_crossover_strategy._prices
    yield
    if hasattr(_sma_crossover_strategy, "_prices"):
        del _sma_crossover_strategy._prices


class TestBacktestResult:
    """Tests for the BacktestResult dataclass."""

    def test_backtest_result_creation(self):
        result = BacktestResult(
            total_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.10,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=20,
            equity_curve=[100000, 101000, 102000],
            trade_log=[{"entry": 100, "exit": 105, "pnl": 500}],
        )
        assert result.total_return == 0.15
        assert result.sharpe_ratio == 1.5
        assert result.total_trades == 20
        assert len(result.equity_curve) == 3
        assert len(result.trade_log) == 1

    def test_backtest_result_defaults(self):
        result = BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            equity_curve=[],
            trade_log=[],
        )
        assert result.total_trades == 0
        assert result.equity_curve == []


class TestBacktestEngine:
    """Tests for the BacktestEngine class."""

    def test_engine_initialization(self):
        engine = BacktestEngine(initial_capital=50000.0, commission=0.002)
        assert engine.initial_capital == 50000.0
        assert engine.commission == 0.002

    def test_engine_default_params(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 100000.0
        assert engine.commission == 0.001

    def test_load_data(self):
        engine = BacktestEngine()
        df = _generate_synthetic_data(100)
        engine.load_data(df)
        assert engine._data is not None
        assert len(engine._data) == 100

    def test_run_sma_crossover(self):
        engine = BacktestEngine(initial_capital=100000.0, commission=0.001)
        df = _generate_synthetic_data(500)
        engine.load_data(df)
        result = engine.run(_sma_crossover_strategy)

        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 100000.0

    def test_run_produces_metrics(self):
        engine = BacktestEngine(initial_capital=100000.0, commission=0.001)
        df = _generate_synthetic_data(500)
        engine.load_data(df)
        result = engine.run(_sma_crossover_strategy)

        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.profit_factor, float)
        assert isinstance(result.total_return, float)

    def test_max_drawdown_is_non_negative(self):
        engine = BacktestEngine()
        df = _generate_synthetic_data(500)
        engine.load_data(df)
        result = engine.run(_sma_crossover_strategy)

        assert result.max_drawdown >= 0.0

    def test_win_rate_in_range(self):
        engine = BacktestEngine()
        df = _generate_synthetic_data(500)
        engine.load_data(df)
        result = engine.run(_sma_crossover_strategy)

        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_length(self):
        engine = BacktestEngine()
        df = _generate_synthetic_data(200)
        engine.load_data(df)
        result = engine.run(_sma_crossover_strategy)

        assert len(result.equity_curve) == 200

    def test_trade_log_structure(self):
        engine = BacktestEngine()
        df = _generate_synthetic_data(500)
        engine.load_data(df)
        result = engine.run(_sma_crossover_strategy)

        if result.total_trades > 0:
            trade = result.trade_log[0]
            assert "entry_price" in trade or "price" in trade or "pnl" in trade

    def test_no_trades_strategy(self):
        """A strategy that never trades should return valid results."""
        engine = BacktestEngine()
        df = _generate_synthetic_data(100)
        engine.load_data(df)

        def do_nothing(index, row, position, capital):
            return 0

        result = engine.run(do_nothing)
        assert result.total_trades == 0
        assert result.total_return == 0.0
        assert result.win_rate == 0.0

    def test_always_buy_strategy(self):
        """A strategy that buys immediately and holds."""
        engine = BacktestEngine()
        df = _generate_synthetic_data(100)
        engine.load_data(df)

        def buy_and_hold(index, row, position, capital):
            if index == 1 and position == 0:
                return 1
            return 0

        result = engine.run(buy_and_hold)
        assert len(result.equity_curve) == 100

    def test_small_dataset(self):
        """Engine should handle very small datasets."""
        engine = BacktestEngine()
        df = _generate_synthetic_data(5)
        engine.load_data(df)

        def do_nothing(index, row, position, capital):
            return 0

        result = engine.run(do_nothing)
        assert result.total_trades == 0
        assert len(result.equity_curve) == 5

    def test_profit_factor_non_negative(self):
        engine = BacktestEngine()
        df = _generate_synthetic_data(500)
        engine.load_data(df)
        result = engine.run(_sma_crossover_strategy)

        if result.total_trades > 0:
            assert result.profit_factor >= 0.0
