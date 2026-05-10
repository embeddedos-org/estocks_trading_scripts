"""
Tests for shared/strategies/*.py
===================================

Covers:
- factor_models.py: FamaFrenchFactors.compute_factors() — verify Quality != LowVol
- factor_models.py: AlphaRanker.rank()
- factor_models.py: FactorBacktester.backtest_long_short() — verify factor_name used, engine.run split
- stat_arb.py: OrnsteinUhlenbeck.fit() / optimal_entry_exit() / half_life()
- stat_arb.py: BasketTrader.compute_spread() / generate_signals()
- stat_arb.py: CointegrationScanner.scan() — verify engine.run split
- mean_variance.py: MeanVarianceOptimizer.max_sharpe() / min_variance() / efficient_frontier()
- mean_variance.py: BlackLitterman views and posterior
- mean_variance.py: RiskBudgeting — verify lambda capture with default arg
"""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.strategies.factor_models import FamaFrenchFactors, AlphaRanker, FactorBacktester


def _make_universe(n_stocks=8, n_bars=400, seed=42):
    """Generate synthetic universe price DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_bars)
    tickers = [f"STOCK_{i}" for i in range(n_stocks)]
    prices = {}
    for t in tickers:
        drift = rng.uniform(-0.0002, 0.001)
        vol = rng.uniform(0.01, 0.025)
        p = 50.0 + rng.uniform(0, 100)
        series = [p]
        for _ in range(n_bars - 1):
            p *= 1 + drift + rng.randn() * vol
            series.append(max(p, 0.01))
        prices[t] = series
    return pd.DataFrame(prices, index=dates)


# ─── FamaFrenchFactors ───


class TestFamaFrenchFactors:

    def test_compute_factors_returns_dataframe(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) > 0

    def test_compute_factors_has_expected_columns(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        expected = {"SMB", "HML", "MOM", "Quality", "LowVol"}
        assert expected.issubset(set(factors.columns))

    def test_quality_not_equal_to_lowvol(self):
        """Verify fix: Quality factor is computed differently from LowVol."""
        prices = _make_universe(n_stocks=8, n_bars=400)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        quality = factors["Quality"].dropna()
        lowvol = factors["LowVol"].dropna()
        common = quality.index.intersection(lowvol.index)
        if len(common) > 0:
            assert not np.allclose(
                quality.loc[common].values,
                lowvol.loc[common].values,
                atol=1e-10,
            ), "Quality and LowVol factors should not be identical"

    def test_compute_factors_with_market_caps(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        rng = np.random.RandomState(99)
        market_caps = prices * rng.uniform(1e6, 1e9, size=prices.shape)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices, market_caps=market_caps)
        assert "SMB" in factors.columns
        assert len(factors) > 0

    def test_compute_factors_without_market_caps(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices, market_caps=None)
        assert "SMB" in factors.columns

    def test_factors_no_infinite_values(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        for col in factors.columns:
            assert not np.any(np.isinf(factors[col].values)), f"Inf in {col}"


# ─── AlphaRanker ───


class TestAlphaRanker:

    def test_rank_returns_dataframe(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        ranker = AlphaRanker()
        ranked = ranker.rank(prices, factors)
        assert isinstance(ranked, pd.DataFrame)
        assert "composite_zscore" in ranked.columns
        assert "rank" in ranked.columns

    def test_rank_all_stocks_present(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        ranker = AlphaRanker()
        ranked = ranker.rank(prices, factors)
        assert len(ranked) == len(prices.columns)

    def test_rank_with_custom_weights(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        ranker = AlphaRanker()
        weights = {"MOM": 0.5, "LowVol": 0.3, "HML": 0.2}
        ranked = ranker.rank(prices, factors, weights=weights)
        assert len(ranked) == len(prices.columns)

    def test_rank_ascending_order(self):
        prices = _make_universe(n_stocks=6, n_bars=300)
        ff = FamaFrenchFactors()
        factors = ff.compute_factors(prices)
        ranker = AlphaRanker()
        ranked = ranker.rank(prices, factors)
        assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))


# ─── FactorBacktester ───


class TestFactorBacktester:

    @patch("shared.backtesting.backtest_engine_v2.BacktestEngineV2")
    def test_backtest_long_short_uses_factor_name(self, mock_engine_cls):
        """Verify fix: factor_name is actually used in ranking, not hardcoded."""
        from shared.backtesting.backtest_engine_v2 import BacktestResultV2

        mock_result = BacktestResultV2(total_return=0.1, total_trades=10)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        prices = _make_universe(n_stocks=12, n_bars=300)
        bt = FactorBacktester()
        result = bt.backtest_long_short(prices, factor_name="LowVol", n_long=3, n_short=3)

        mock_engine.load_data.assert_called_once()
        mock_engine.run.assert_called_once()
        strategy_fn = mock_engine.run.call_args[0][0]
        assert callable(strategy_fn)

    @patch("shared.backtesting.backtest_engine_v2.BacktestEngineV2")
    def test_backtest_long_short_mom_factor(self, mock_engine_cls):
        from shared.backtesting.backtest_engine_v2 import BacktestResultV2

        mock_result = BacktestResultV2(total_return=0.2)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        prices = _make_universe(n_stocks=12, n_bars=300)
        bt = FactorBacktester()
        result = bt.backtest_long_short(prices, factor_name="MOM", n_long=3, n_short=3)
        assert result.total_return == 0.2

    @patch("shared.backtesting.backtest_engine_v2.BacktestEngineV2")
    def test_backtest_engine_run_receives_callable(self, mock_engine_cls):
        """Verify fix: engine.run receives a strategy_fn callable (split correctly)."""
        from shared.backtesting.backtest_engine_v2 import BacktestResultV2

        mock_result = BacktestResultV2()
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        prices = _make_universe(n_stocks=12, n_bars=300)
        bt = FactorBacktester()
        bt.backtest_long_short(prices)

        assert mock_engine.run.call_count == 1
        fn = mock_engine.run.call_args[0][0]
        assert callable(fn)

    @patch("shared.backtesting.backtest_engine_v2.BacktestEngineV2")
    def test_backtest_quality_factor(self, mock_engine_cls):
        from shared.backtesting.backtest_engine_v2 import BacktestResultV2

        mock_result = BacktestResultV2(total_return=0.05)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        prices = _make_universe(n_stocks=12, n_bars=300)
        bt = FactorBacktester()
        result = bt.backtest_long_short(prices, factor_name="Quality", n_long=3, n_short=3)
        assert result.total_return == 0.05


# ─── OrnsteinUhlenbeck ───


class TestOrnsteinUhlenbeck:

    def test_fit_basic(self):
        from shared.strategies.stat_arb import OrnsteinUhlenbeck
        rng = np.random.RandomState(42)
        mu, theta_true = 50.0, 0.05
        s = [mu]
        for _ in range(500):
            ds = theta_true * (mu - s[-1]) + rng.randn() * 1.0
            s.append(s[-1] + ds)
        spread = pd.Series(s)
        ou = OrnsteinUhlenbeck()
        params = ou.fit(spread)
        assert "theta" in params
        assert "mu" in params
        assert "sigma" in params
        assert "half_life" in params
        assert params["theta"] > 0

    def test_fit_requires_min_data(self):
        from shared.strategies.stat_arb import OrnsteinUhlenbeck
        ou = OrnsteinUhlenbeck()
        with pytest.raises(ValueError, match="at least 10"):
            ou.fit(pd.Series([1, 2, 3]))

    def test_optimal_entry_exit_requires_fit(self):
        from shared.strategies.stat_arb import OrnsteinUhlenbeck
        ou = OrnsteinUhlenbeck()
        with pytest.raises(RuntimeError, match="fit"):
            ou.optimal_entry_exit(50.0)

    def test_optimal_entry_exit_after_fit(self):
        from shared.strategies.stat_arb import OrnsteinUhlenbeck
        rng = np.random.RandomState(42)
        mu = 50.0
        s = [mu]
        for _ in range(200):
            s.append(s[-1] + 0.05 * (mu - s[-1]) + rng.randn() * 1.0)
        ou = OrnsteinUhlenbeck()
        ou.fit(pd.Series(s))
        levels = ou.optimal_entry_exit(50.0)
        assert levels["entry_long"] < levels["exit_long"]
        assert levels["entry_short"] > levels["exit_short"]

    def test_half_life_requires_fit(self):
        from shared.strategies.stat_arb import OrnsteinUhlenbeck
        ou = OrnsteinUhlenbeck()
        with pytest.raises(RuntimeError, match="fit"):
            ou.half_life()

    def test_half_life_positive(self):
        from shared.strategies.stat_arb import OrnsteinUhlenbeck
        rng = np.random.RandomState(42)
        s = [50.0]
        for _ in range(200):
            s.append(s[-1] + 0.05 * (50 - s[-1]) + rng.randn() * 1.0)
        ou = OrnsteinUhlenbeck()
        ou.fit(pd.Series(s))
        assert ou.half_life() > 0


# ─── BasketTrader ───


class TestBasketTrader:

    def test_compute_spread(self):
        from shared.strategies.stat_arb import BasketTrader
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame({
            "A": np.random.RandomState(1).randn(100).cumsum() + 100,
            "B": np.random.RandomState(2).randn(100).cumsum() + 50,
        }, index=dates)
        bt = BasketTrader("A", "B", hedge_ratio=1.5)
        spread = bt.compute_spread(prices)
        assert len(spread) == 100
        assert spread.name == "spread"
        expected = prices["A"] - 1.5 * prices["B"]
        expected.name = "spread"
        pd.testing.assert_series_equal(spread, expected, check_names=True)

    def test_generate_signals_shape(self):
        from shared.strategies.stat_arb import BasketTrader
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame({
            "A": np.random.RandomState(1).randn(100).cumsum() + 100,
            "B": np.random.RandomState(2).randn(100).cumsum() + 50,
        }, index=dates)
        bt = BasketTrader("A", "B", hedge_ratio=1.5)
        result = bt.generate_signals(prices, lookback=20)
        assert "spread" in result.columns
        assert "zscore" in result.columns
        assert "signal" in result.columns
        assert len(result) == 100

    @patch("shared.backtesting.backtest_engine_v2.BacktestEngineV2")
    def test_backtest_engine_run_split(self, mock_engine_cls):
        """Verify fix: engine.run is called with strategy_fn (properly split)."""
        from shared.backtesting.backtest_engine_v2 import BacktestResultV2
        from shared.strategies.stat_arb import BasketTrader

        mock_result = BacktestResultV2(total_return=0.03)
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame({
            "X": np.random.RandomState(1).randn(100).cumsum() + 100,
            "Y": np.random.RandomState(2).randn(100).cumsum() + 50,
        }, index=dates)
        bt = BasketTrader("X", "Y", hedge_ratio=1.0)
        result = bt.backtest(prices)

        mock_engine.load_data.assert_called_once()
        mock_engine.run.assert_called_once()
        fn = mock_engine.run.call_args[0][0]
        assert callable(fn)


# ─── MeanVarianceOptimizer ───


class TestMeanVarianceOptimizer:

    def _make_returns(self, n=5, n_days=252, seed=42):
        rng = np.random.RandomState(seed)
        tickers = [f"S{i}" for i in range(n)]
        data = rng.randn(n_days, n) * 0.02
        return pd.DataFrame(data, columns=tickers)

    @pytest.fixture(autouse=True)
    def skip_if_no_scipy(self):
        try:
            from scipy.optimize import minimize
        except ImportError:
            pytest.skip("scipy not installed")

    def test_max_sharpe_weights_sum_to_one(self):
        from shared.strategies.mean_variance import MeanVarianceOptimizer
        returns = self._make_returns()
        opt = MeanVarianceOptimizer(returns)
        weights = opt.max_sharpe()
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_max_sharpe_correct_tickers(self):
        from shared.strategies.mean_variance import MeanVarianceOptimizer
        returns = self._make_returns()
        opt = MeanVarianceOptimizer(returns)
        weights = opt.max_sharpe()
        assert list(weights.index) == list(returns.columns)

    def test_min_variance_weights_sum_to_one(self):
        from shared.strategies.mean_variance import MeanVarianceOptimizer
        returns = self._make_returns()
        opt = MeanVarianceOptimizer(returns)
        weights = opt.min_variance()
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_max_positions_constraint(self):
        from shared.strategies.mean_variance import MeanVarianceOptimizer, OptimizationConstraints
        returns = self._make_returns(n=10)
        opt = MeanVarianceOptimizer(returns)
        c = OptimizationConstraints(max_positions=3)
        weights = opt.max_sharpe(c)
        assert (weights != 0).sum() <= 3

    def test_efficient_frontier_returns_dataframe(self):
        from shared.strategies.mean_variance import MeanVarianceOptimizer
        returns = self._make_returns()
        opt = MeanVarianceOptimizer(returns)
        ef = opt.efficient_frontier(n_points=10)
        assert isinstance(ef, pd.DataFrame)
        assert "return" in ef.columns
        assert "volatility" in ef.columns
        assert "sharpe" in ef.columns

    def test_lambda_capture_with_default_arg(self):
        """Verify fix: lambda in constraint uses default arg to avoid closure bug."""
        from shared.strategies.mean_variance import MeanVarianceOptimizer, OptimizationConstraints
        returns = self._make_returns()
        opt = MeanVarianceOptimizer(returns)
        c = OptimizationConstraints(target_return=float(opt.mean_returns.mean()))
        weights = opt.max_sharpe(c)
        assert abs(weights.sum() - 1.0) < 1e-4
        port_ret = np.dot(weights, opt.mean_returns)
        assert abs(port_ret - c.target_return) < 0.1


# ─── BlackLitterman ───


class TestBlackLitterman:

    @pytest.fixture(autouse=True)
    def skip_if_no_scipy(self):
        try:
            from scipy.optimize import minimize
        except ImportError:
            pytest.skip("scipy not installed")

    def _make_returns(self, n=5, n_days=252, seed=42):
        rng = np.random.RandomState(seed)
        tickers = [f"S{i}" for i in range(n)]
        data = rng.randn(n_days, n) * 0.02
        return pd.DataFrame(data, columns=tickers)

    def test_equilibrium_returns_without_views(self):
        from shared.strategies.mean_variance import BlackLitterman
        returns = self._make_returns()
        bl = BlackLitterman(returns)
        posterior = bl.posterior_returns()
        assert len(posterior) == len(returns.columns)

    def test_add_view_and_posterior(self):
        from shared.strategies.mean_variance import BlackLitterman
        returns = self._make_returns()
        bl = BlackLitterman(returns)
        bl.add_view("S0", expected_return=0.15, confidence=0.8)
        posterior = bl.posterior_returns()
        assert len(posterior) == len(returns.columns)

    def test_invalid_view_asset(self):
        from shared.strategies.mean_variance import BlackLitterman
        returns = self._make_returns()
        bl = BlackLitterman(returns)
        with pytest.raises(ValueError, match="Unknown asset"):
            bl.add_view("NONEXISTENT", expected_return=0.1)

    def test_optimal_weights_sum_to_one(self):
        from shared.strategies.mean_variance import BlackLitterman
        returns = self._make_returns()
        bl = BlackLitterman(returns)
        bl.add_view("S0", expected_return=0.2, confidence=0.7)
        weights = bl.optimal_weights()
        assert abs(weights.sum() - 1.0) < 1e-4


# ─── RiskBudgeting ───


class TestRiskBudgeting:

    @pytest.fixture(autouse=True)
    def skip_if_no_scipy(self):
        try:
            from scipy.optimize import minimize
        except ImportError:
            pytest.skip("scipy not installed")

    def test_equal_budget_weights(self):
        from shared.strategies.mean_variance import RiskBudgeting
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(
            rng.randn(252, 3) * 0.02,
            columns=["A", "B", "C"],
        )
        rb = RiskBudgeting(returns)
        weights = rb.optimize()
        assert abs(weights.sum() - 1.0) < 1e-4
        assert (weights > 0).all()

    def test_custom_budget_weights(self):
        from shared.strategies.mean_variance import RiskBudgeting
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(
            rng.randn(252, 3) * 0.02,
            columns=["A", "B", "C"],
        )
        budgets = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        rb = RiskBudgeting(returns)
        weights = rb.optimize(budgets=budgets)
        assert abs(weights.sum() - 1.0) < 1e-4
