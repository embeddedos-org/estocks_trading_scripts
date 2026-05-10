"""
Tests for strategies/examples/*.py
======================================

Covers:
- TrendFollowingStrategy: generate_signals() with trending/flat/volatile data
- MeanReversionStrategy: generate_signals() with oversold/overbought/normal data
- BreakoutStrategy: generate_signals() with breakout and consolidation data
- SelfLearningStrategy: verify fix — internal entry_price tracking, PnL non-zero
- MLRLStrategy: verify fix — features passed to predictor, not raw df
- FactorPortfolioStrategy: verify fix — sorted with lambda key
"""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.backtesting.backtest_engine_v2 import BacktestContext, BacktestResultV2
from strategies.examples.trend_following import TrendFollowingStrategy, TrendFollowingConfig
from strategies.examples.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from strategies.examples.breakout import BreakoutStrategy, BreakoutConfig
from strategies.examples.self_learning_strategy import SelfLearningStrategy, SelfLearningConfig
from strategies.examples.ml_rl_strategy import MLStrategy, MLConfig, RLStrategy, RLConfig
from strategies.examples.factor_portfolio import FactorPortfolioStrategy, FactorPortfolioConfig


def _make_ohlcv(n=300, seed=42, drift=0.0, volatility=0.015, start_price=100.0):
    """Helper to generate synthetic OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = start_price
    rows = []
    for i in range(n):
        ret = drift + rng.randn() * volatility
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.005)
        low = price * (1 - abs(rng.randn()) * 0.005)
        rows.append({
            "date": dates[i],
            "open": price * (1 + rng.randn() * 0.002),
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(rows)


def _make_context(df, positions=None, bar_index=0):
    """Create a BacktestContext from a single-stock DataFrame."""
    return BacktestContext(
        bar_index=bar_index,
        bars={"TEST": df},
        positions=positions or {},
        capital=100_000.0,
        portfolio_value=100_000.0,
    )


# ─── TrendFollowingStrategy ───


class TestTrendFollowingStrategy:

    def test_default_config(self):
        s = TrendFollowingStrategy()
        assert s.config.fast_ma_length == 9
        assert s.config.slow_ma_length == 21
        assert s.config.use_adx_filter is True

    def test_from_params(self):
        s = TrendFollowingStrategy.from_params({"fast_ma_length": 5, "adx_threshold": 20})
        assert s.config.fast_ma_length == 5
        assert s.config.adx_threshold == 20

    def test_signal_with_insufficient_data(self):
        df = _make_ohlcv(n=50)
        ctx = _make_context(df)
        s = TrendFollowingStrategy()
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_signal_with_trending_data(self):
        df = _make_ohlcv(n=300, drift=0.002)
        ctx = _make_context(df)
        s = TrendFollowingStrategy(TrendFollowingConfig(use_adx_filter=False))
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_signal_with_flat_data(self):
        df = _make_ohlcv(n=300, drift=0.0, volatility=0.001)
        ctx = _make_context(df)
        s = TrendFollowingStrategy()
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_signal_with_volatile_data(self):
        df = _make_ohlcv(n=300, drift=0.0, volatility=0.05)
        ctx = _make_context(df)
        s = TrendFollowingStrategy()
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_trailing_stop_long_triggers_exit(self):
        df = _make_ohlcv(n=300, drift=0.001)
        s = TrendFollowingStrategy(TrendFollowingConfig(trailing_stop=True, use_adx_filter=False))
        s._trailing_stops["TEST"] = float(df["close"].iloc[-1]) + 1000
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_trailing_stop_short_triggers_exit(self):
        df = _make_ohlcv(n=300, drift=-0.001)
        s = TrendFollowingStrategy(TrendFollowingConfig(trailing_stop=True, use_adx_filter=False))
        s._trailing_stops["TEST"] = float(df["close"].iloc[-1]) - 1000
        ctx = _make_context(df, positions={"TEST": -100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_hold_existing_position(self):
        df = _make_ohlcv(n=300, drift=0.002)
        s = TrendFollowingStrategy(TrendFollowingConfig(use_adx_filter=False))
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)
        assert isinstance(signals["TEST"], int)


# ─── MeanReversionStrategy ───


class TestMeanReversionStrategy:

    def test_default_config(self):
        s = MeanReversionStrategy()
        assert s.config.rsi_length == 14
        assert s.config.rsi_oversold == 30
        assert s.config.rsi_overbought == 70

    def test_from_params(self):
        s = MeanReversionStrategy.from_params({"rsi_oversold": 25, "bb_std": 2.5})
        assert s.config.rsi_oversold == 25
        assert s.config.bb_std == 2.5

    def test_signal_with_insufficient_data(self):
        df = _make_ohlcv(n=10)
        ctx = _make_context(df)
        s = MeanReversionStrategy()
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_signal_with_normal_data(self):
        df = _make_ohlcv(n=200)
        ctx = _make_context(df)
        s = MeanReversionStrategy()
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_stop_loss_triggers_exit(self):
        df = _make_ohlcv(n=200)
        s = MeanReversionStrategy(MeanReversionConfig(stop_loss_pct=0.001))
        s._entry_prices["TEST"] = float(df["close"].iloc[-1]) * 1.5
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_exit_at_bb_midline_long(self):
        df = _make_ohlcv(n=200)
        s = MeanReversionStrategy()
        s._entry_prices["TEST"] = float(df["close"].iloc[-1]) * 0.5
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {0, 1}

    def test_hold_existing_long(self):
        df = _make_ohlcv(n=200)
        s = MeanReversionStrategy()
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_without_bb_confirm(self):
        df = _make_ohlcv(n=200)
        s = MeanReversionStrategy(MeanReversionConfig(use_bb_confirm=False))
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}


# ─── BreakoutStrategy ───


class TestBreakoutStrategy:

    def test_default_config(self):
        s = BreakoutStrategy()
        assert s.config.channel_length == 20
        assert s.config.volume_mult == 1.5
        assert s.config.confirm_bars == 1

    def test_from_params(self):
        s = BreakoutStrategy.from_params({"channel_length": 30, "confirm_bars": 2})
        assert s.config.channel_length == 30
        assert s.config.confirm_bars == 2

    def test_signal_with_insufficient_data(self):
        df = _make_ohlcv(n=10)
        ctx = _make_context(df)
        s = BreakoutStrategy()
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_signal_with_consolidation_data(self):
        df = _make_ohlcv(n=100, drift=0.0, volatility=0.003)
        ctx = _make_context(df)
        s = BreakoutStrategy()
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_trailing_stop_long_exit(self):
        df = _make_ohlcv(n=100)
        s = BreakoutStrategy()
        s._trailing_stops["TEST"] = float(df["close"].iloc[-1]) + 1000
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_trailing_stop_short_exit(self):
        df = _make_ohlcv(n=100)
        s = BreakoutStrategy()
        s._trailing_stops["TEST"] = float(df["close"].iloc[-1]) - 1000
        ctx = _make_context(df, positions={"TEST": -100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_hold_long_position(self):
        df = _make_ohlcv(n=100)
        s = BreakoutStrategy()
        s._trailing_stops["TEST"] = 0.0
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {0, 1}

    def test_breakout_confirmation_bars_config(self):
        s = BreakoutStrategy(BreakoutConfig(confirm_bars=3))
        assert s.config.confirm_bars == 3


# ─── SelfLearningStrategy ───


class TestSelfLearningStrategy:

    @patch("shared.ml.self_learning_agent.SelfLearningAgent", side_effect=ImportError("no deps"))
    def test_fallback_mode_on_import_error(self, mock_agent):
        s = SelfLearningStrategy()
        assert s._fallback is True

    def test_train_fallback_skips(self):
        s = SelfLearningStrategy.__new__(SelfLearningStrategy)
        s.config = SelfLearningConfig()
        s._fallback = True
        s._agent = None
        s._trained = False
        s._entry_prices = {}
        df = _make_ohlcv(n=100)
        s.train(df)

    def test_generate_signals_fallback_momentum(self):
        s = SelfLearningStrategy.__new__(SelfLearningStrategy)
        s.config = SelfLearningConfig()
        s._fallback = True
        s._agent = None
        s._trained = False
        s._entry_prices = {}

        df = _make_ohlcv(n=100, drift=0.005)
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_momentum_signal_bullish(self):
        s = SelfLearningStrategy.__new__(SelfLearningStrategy)
        s.config = SelfLearningConfig()
        s._fallback = True
        s._agent = None
        s._trained = False
        s._entry_prices = {}

        prices = [100 + i * 2 for i in range(20)]
        df = pd.DataFrame({
            "close": prices, "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": [1_000_000] * 20,
        })
        signal = s._momentum_signal(df)
        assert signal == 1

    def test_momentum_signal_bearish(self):
        s = SelfLearningStrategy.__new__(SelfLearningStrategy)
        s.config = SelfLearningConfig()
        s._fallback = True
        s._agent = None
        s._trained = False
        s._entry_prices = {}

        prices = [200 - i * 2 for i in range(20)]
        df = pd.DataFrame({
            "close": prices, "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": [1_000_000] * 20,
        })
        signal = s._momentum_signal(df)
        assert signal == -1

    def test_entry_price_tracking_on_buy(self):
        """Verify fix: internal entry_price is stored when BUY signal fires."""
        mock_agent = MagicMock()
        mock_agent.decide.return_value = {"action": "BUY", "confidence": 0.8}

        s = SelfLearningStrategy.__new__(SelfLearningStrategy)
        s.config = SelfLearningConfig()
        s._agent = mock_agent
        s._fallback = False
        s._trained = True
        s._entry_prices = {}

        df = _make_ohlcv(n=100)
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 1
        assert "TEST" in s._entry_prices
        assert s._entry_prices["TEST"] == float(df["close"].iloc[-1])

    def test_pnl_recorded_on_position_change(self):
        """Verify fix: record_outcome is called when position flips."""
        mock_agent = MagicMock()
        mock_agent.decide.return_value = {"action": "SELL", "confidence": 0.8}

        s = SelfLearningStrategy.__new__(SelfLearningStrategy)
        s.config = SelfLearningConfig()
        s._agent = mock_agent
        s._fallback = False
        s._trained = True
        s._entry_prices = {"TEST": 50.0}

        df = _make_ohlcv(n=100)
        ctx = _make_context(df, positions={"TEST": 100})
        signals = s.generate_signals(ctx)

        mock_agent.record_outcome.assert_called_once()
        call_kwargs = mock_agent.record_outcome.call_args[1]
        assert "pnl" in call_kwargs
        assert "exit_price" in call_kwargs

    def test_insufficient_data_returns_zero(self):
        s = SelfLearningStrategy.__new__(SelfLearningStrategy)
        s.config = SelfLearningConfig()
        s._fallback = True
        s._agent = None
        s._trained = False
        s._entry_prices = {}

        df = _make_ohlcv(n=30)
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0


# ─── MLStrategy ───


class TestMLStrategy:

    def test_fallback_momentum_signal(self):
        s = MLStrategy.__new__(MLStrategy)
        s.config = MLConfig()
        s._predictor = None
        s._feature_engineer = None
        s._fallback = True
        s._entry_prices = {}
        s._peak_prices = {}

        df = _make_ohlcv(n=100, drift=0.003)
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_train_fallback_prints_message(self, capsys):
        s = MLStrategy.__new__(MLStrategy)
        s.config = MLConfig()
        s._predictor = None
        s._feature_engineer = None
        s._fallback = True
        s._entry_prices = {}
        s._peak_prices = {}

        df = _make_ohlcv(n=100)
        s.train(df)
        captured = capsys.readouterr()
        assert "fallback" in captured.out.lower() or "momentum" in captured.out.lower()

    def test_predictor_receives_features_not_raw_df(self):
        """Verify fix: features (not raw DataFrame) are passed to predictor."""
        mock_fe = MagicMock()
        mock_predictor = MagicMock()

        features_df = pd.DataFrame(
            {"f1": [1, 2, 3], "f2": [4, 5, 6]},
            index=[10, 11, 12],
        )
        mock_fe.compute_features.return_value = features_df

        s = MLStrategy.__new__(MLStrategy)
        s.config = MLConfig()
        s._feature_engineer = mock_fe
        s._predictor = mock_predictor
        s._fallback = False
        s._entry_prices = {}
        s._peak_prices = {}

        raw_df = _make_ohlcv(n=300)
        s.train(raw_df)

        mock_fe.compute_features.assert_called_once_with(raw_df)
        train_call = mock_predictor.train.call_args
        passed_features = train_call[0][0]
        assert isinstance(passed_features, pd.DataFrame)
        assert "close" not in passed_features.columns

    def test_predict_exception_returns_zero(self):
        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = RuntimeError("model error")

        s = MLStrategy.__new__(MLStrategy)
        s.config = MLConfig()
        s._predictor = mock_predictor
        s._feature_engineer = None
        s._fallback = False
        s._entry_prices = {}
        s._peak_prices = {}

        df = _make_ohlcv(n=100)
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0

    def test_from_params(self):
        s = MLStrategy.from_params({"seq_len": 30, "epochs": 10})
        assert s.config.seq_len == 30
        assert s.config.epochs == 10


# ─── RLStrategy ───


class TestRLStrategy:

    def test_fallback_rsi_signal(self):
        s = RLStrategy.__new__(RLStrategy)
        s.config = RLConfig()
        s._trader = None
        s._fallback = True
        s._entry_prices = {}
        s._peak_prices = {}

        df = _make_ohlcv(n=100)
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] in {-1, 0, 1}

    def test_train_fallback(self, capsys):
        s = RLStrategy.__new__(RLStrategy)
        s.config = RLConfig()
        s._trader = None
        s._fallback = True
        s._entry_prices = {}
        s._peak_prices = {}

        df = _make_ohlcv(n=100)
        s.train(df)
        captured = capsys.readouterr()
        assert "fallback" in captured.out.lower() or "RSI" in captured.out

    def test_from_params(self):
        s = RLStrategy.from_params({"algorithm": "A2C", "total_timesteps": 10000})
        assert s.config.algorithm == "A2C"
        assert s.config.total_timesteps == 10000

    def test_insufficient_data(self):
        s = RLStrategy.__new__(RLStrategy)
        s.config = RLConfig()
        s._trader = None
        s._fallback = True
        s._entry_prices = {}
        s._peak_prices = {}

        df = _make_ohlcv(n=10)
        ctx = _make_context(df)
        signals = s.generate_signals(ctx)
        assert signals["TEST"] == 0


# ─── FactorPortfolioStrategy ───


class TestFactorPortfolioStrategy:

    def test_default_config(self):
        s = FactorPortfolioStrategy()
        assert s.config.n_long == 5
        assert s.config.n_short == 5
        assert s.config.rebalance_freq == 21

    def test_from_params(self):
        s = FactorPortfolioStrategy.from_params({"n_long": 3, "n_short": 3})
        assert s.config.n_long == 3
        assert s.config.n_short == 3

    def test_sorted_with_lambda_key(self):
        """Verify fix: sorted() uses lambda key=lambda k: momentum_scores.get(k, 0.0)."""
        s = FactorPortfolioStrategy(FactorPortfolioConfig(
            n_long=2, n_short=2, momentum_lookback=50, momentum_skip=5,
        ))
        n_bars = 100
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=n_bars)

        tickers = ["A", "B", "C", "D", "E"]
        bars = {}
        for i, t in enumerate(tickers):
            drift = 0.005 * (i - 2)
            price = 100.0
            rows = []
            for j in range(n_bars):
                price *= 1 + drift + rng.randn() * 0.01
                rows.append({
                    "date": dates[j], "open": price, "high": price * 1.01,
                    "low": price * 0.99, "close": price, "volume": 1_000_000,
                })
            bars[t] = pd.DataFrame(rows)

        s._tickers = tickers
        ctx = BacktestContext(
            bar_index=0,
            bars=bars,
            positions={},
            capital=100_000.0,
            portfolio_value=100_000.0,
        )
        signals = s.generate_signals(ctx)
        long_count = sum(1 for v in signals.values() if v == 1)
        short_count = sum(1 for v in signals.values() if v == -1)
        assert long_count == 2
        assert short_count == 2

    def test_maintain_positions_between_rebalances(self):
        s = FactorPortfolioStrategy(FactorPortfolioConfig(rebalance_freq=21))
        s._tickers = ["A", "B"]
        bars = {
            t: _make_ohlcv(n=300) for t in s._tickers
        }
        ctx = BacktestContext(
            bar_index=5,
            bars=bars,
            positions={"A": 100, "B": -50},
            capital=100_000.0,
            portfolio_value=100_000.0,
        )
        signals = s.generate_signals(ctx)
        assert signals["A"] == 1
        assert signals["B"] == -1

    def test_insufficient_history_returns_zeros(self):
        s = FactorPortfolioStrategy(FactorPortfolioConfig(
            momentum_lookback=252, momentum_skip=21,
        ))
        s._tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        bars = {t: _make_ohlcv(n=50) for t in s._tickers}
        ctx = BacktestContext(
            bar_index=0, bars=bars, positions={},
            capital=100_000.0, portfolio_value=100_000.0,
        )
        signals = s.generate_signals(ctx)
        assert all(v == 0 for v in signals.values())

    @patch("strategies.examples.factor_portfolio.BacktestEngineV2")
    def test_run_backtest_creates_engine(self, mock_engine_cls):
        mock_result = BacktestResultV2(total_return=0.05, total_trades=10)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2019-01-01", periods=300)
        universe = pd.DataFrame(
            {f"S{i}": 100 + rng.randn(300).cumsum() for i in range(5)},
            index=dates,
        )
        s = FactorPortfolioStrategy(FactorPortfolioConfig(n_long=2, n_short=2))
        result = s.run_backtest(universe)

        assert result.total_return == 0.05
        mock_engine.load_data.assert_called_once()
        mock_engine.run.assert_called_once()
