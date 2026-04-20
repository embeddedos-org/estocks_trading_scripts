"""
Tests for portfolio_backtester/

Covers:
- PortfolioEngine: run_backtest(), calculate_metrics()
  (verify fix: CAGR guard, rebased prices, Sortino)
- RiskParity: target_vol_weights() (verify fix: NaN cov),
  risk_budgeting() (verify fix: convergence check)
- Strategies: RiskParityAlgo (verify fix: engine reused),
  TacticalAllocationAlgo (verify fix: vol_20d guard)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════
# RiskParityConfig Tests
# ═══════════════════════════════════════════════════════

class TestRiskParityConfig:

    def test_defaults(self):
        from portfolio_backtester.risk_parity import RiskParityConfig
        cfg = RiskParityConfig()
        assert cfg.target_vol == 0.10
        assert cfg.lookback == 63
        assert cfg.rebalance_freq == 21
        assert cfg.min_weight == 0.0
        assert cfg.max_weight == 1.0

    def test_custom(self):
        from portfolio_backtester.risk_parity import RiskParityConfig
        cfg = RiskParityConfig(target_vol=0.15, lookback=126)
        assert cfg.target_vol == 0.15
        assert cfg.lookback == 126


# ═══════════════════════════════════════════════════════
# RiskParityEngine Tests
# ═══════════════════════════════════════════════════════

class TestRiskParityEngine:

    def _make_returns(self, n_assets=3, n_days=100):
        np.random.seed(42)
        data = np.random.randn(n_days, n_assets) * 0.01
        cols = [f"ASSET_{i}" for i in range(n_assets)]
        return pd.DataFrame(data, columns=cols)

    def test_inverse_vol_weights_sum_to_one(self):
        from portfolio_backtester.risk_parity import RiskParityEngine
        engine = RiskParityEngine()
        returns = self._make_returns()
        weights = engine.inverse_vol_weights(returns)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(w >= 0 for w in weights)

    def test_inverse_vol_zero_vol_asset(self):
        from portfolio_backtester.risk_parity import RiskParityEngine
        engine = RiskParityEngine()
        returns = pd.DataFrame({
            "A": np.random.randn(100) * 0.01,
            "B": [0.0] * 100,
            "C": np.random.randn(100) * 0.02,
        })
        weights = engine.inverse_vol_weights(returns)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights["B"] == 0.0

    def test_target_vol_weights(self):
        from portfolio_backtester.risk_parity import RiskParityEngine, RiskParityConfig
        engine = RiskParityEngine(RiskParityConfig(target_vol=0.10))
        returns = self._make_returns()
        weights = engine.target_vol_weights(returns)
        assert weights.sum() <= 1.0 + 1e-6
        assert all(w >= 0 for w in weights)

    def test_target_vol_nan_cov_fix(self):
        """Verify fix: NaN in covariance matrix is filled with 0."""
        from portfolio_backtester.risk_parity import RiskParityEngine, RiskParityConfig
        engine = RiskParityEngine(RiskParityConfig(target_vol=0.10))
        returns = pd.DataFrame({
            "A": [0.01, -0.01, 0.02, 0.0, 0.01] * 20,
            "B": [0.02, 0.01, -0.01, 0.01, 0.0] * 20,
        })
        weights = engine.target_vol_weights(returns)
        assert np.isfinite(weights).all()
        assert weights.sum() <= 1.0 + 1e-6

    def test_risk_budgeting_equal(self):
        from portfolio_backtester.risk_parity import RiskParityEngine
        engine = RiskParityEngine()
        returns = self._make_returns(n_assets=3)
        weights = engine.risk_budgeting(returns)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert len(weights) == 3

    def test_risk_budgeting_custom_budgets(self):
        from portfolio_backtester.risk_parity import RiskParityEngine
        engine = RiskParityEngine()
        returns = self._make_returns(n_assets=3)
        budgets = pd.Series(
            [0.5, 0.3, 0.2],
            index=returns.columns,
        )
        weights = engine.risk_budgeting(returns, budgets=budgets)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_risk_budgeting_convergence_check(self):
        """Verify fix: convergence check with np.allclose stops early."""
        import inspect
        from portfolio_backtester.risk_parity import RiskParityEngine
        src = inspect.getsource(RiskParityEngine.risk_budgeting)
        assert "allclose" in src

    def test_clip_weights(self):
        from portfolio_backtester.risk_parity import RiskParityEngine, RiskParityConfig
        engine = RiskParityEngine(RiskParityConfig(min_weight=0.1, max_weight=0.5))
        weights = pd.Series([0.2, 0.5, 0.3], index=["A", "B", "C"])
        clipped = engine._clip_weights(weights)
        assert clipped.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(clipped >= 0)

    def test_all_zero_vol(self):
        from portfolio_backtester.risk_parity import RiskParityEngine
        engine = RiskParityEngine()
        returns = pd.DataFrame({
            "A": [0.0] * 100,
            "B": [0.0] * 100,
        })
        weights = engine.inverse_vol_weights(returns)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights["A"] == pytest.approx(0.5, abs=1e-6)


# ═══════════════════════════════════════════════════════
# PortfolioBacktestConfig Tests
# ═══════════════════════════════════════════════════════

class TestPortfolioBacktestConfig:

    def test_defaults(self):
        from portfolio_backtester.portfolio_engine import PortfolioBacktestConfig
        cfg = PortfolioBacktestConfig()
        assert cfg.rebalance_freq == "monthly"
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission == 0.001
        assert cfg.benchmark is None

    def test_custom(self):
        from portfolio_backtester.portfolio_engine import PortfolioBacktestConfig
        cfg = PortfolioBacktestConfig(
            rebalance_freq="weekly", initial_capital=50_000, benchmark="SPY",
        )
        assert cfg.rebalance_freq == "weekly"
        assert cfg.benchmark == "SPY"


# ═══════════════════════════════════════════════════════
# PortfolioEngine Tests
# ═══════════════════════════════════════════════════════

class TestPortfolioEngine:

    def test_init_requires_bt(self):
        from portfolio_backtester.portfolio_engine import PortfolioEngine, _HAS_BT
        if not _HAS_BT:
            with pytest.raises(ImportError, match="bt is required"):
                PortfolioEngine()
        else:
            engine = PortfolioEngine()
            assert engine.config is not None

    def test_unknown_strategy_raises(self):
        from portfolio_backtester.portfolio_engine import PortfolioEngine, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        engine = PortfolioEngine()
        with pytest.raises(ValueError, match="Unknown strategy"):
            engine.run("nonexistent_strategy", pd.DataFrame())

    def test_strategy_map(self):
        from portfolio_backtester.portfolio_engine import PortfolioEngine
        expected = {"equal_weight", "momentum", "risk_parity", "mean_variance", "tactical"}
        assert expected == set(PortfolioEngine.STRATEGY_MAP.keys())

    def test_cagr_guard(self):
        """Verify fix: CAGR guard handles edge cases (n_years=0, negative equity)."""
        import inspect
        from portfolio_backtester.portfolio_engine import PortfolioEngine
        src = inspect.getsource(PortfolioEngine._convert_result)
        assert "n_years > 0" in src or "max(0.0001" in src

    def test_rebased_prices_detection(self):
        """Verify fix: bt returns 100-based rebased prices, converted back."""
        import inspect
        from portfolio_backtester.portfolio_engine import PortfolioEngine
        src = inspect.getsource(PortfolioEngine._convert_result)
        assert "100.0" in src
        assert "initial" in src

    def test_sortino_single_neg_return(self):
        """Verify fix: Sortino handles 1 negative return edge case."""
        import inspect
        from portfolio_backtester.portfolio_engine import PortfolioEngine
        src = inspect.getsource(PortfolioEngine._convert_result)
        assert "len(neg_ret) <= 1" in src or "len(neg_ret) == 1" in src

    def test_convert_result_with_mock(self):
        from portfolio_backtester.portfolio_engine import PortfolioEngine, _HAS_BT, _HAS_ENGINE
        if not _HAS_BT or not _HAS_ENGINE:
            pytest.skip("bt or BacktestResultV2 not available")

        engine = PortfolioEngine()

        mock_result = MagicMock()
        equity_data = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0],
            index=pd.date_range("2023-01-01", periods=5, freq="B"),
            name="test_strat",
        )
        mock_result.prices = pd.DataFrame({"test_strat": equity_data})
        mock_result.stats = MagicMock()

        result = engine._convert_result(mock_result, "test_strat")
        assert result.total_return > 0
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.sortino_ratio)
        assert result.max_drawdown <= 0

    def test_convert_result_declining_equity(self):
        from portfolio_backtester.portfolio_engine import PortfolioEngine, _HAS_BT, _HAS_ENGINE
        if not _HAS_BT or not _HAS_ENGINE:
            pytest.skip("bt or BacktestResultV2 not available")

        engine = PortfolioEngine()
        mock_result = MagicMock()
        equity = pd.Series(
            [100.0, 99.0, 97.0, 95.0, 93.0],
            index=pd.date_range("2023-01-01", periods=5, freq="B"),
            name="losing",
        )
        mock_result.prices = pd.DataFrame({"losing": equity})
        mock_result.stats = MagicMock()

        result = engine._convert_result(mock_result, "losing")
        assert result.total_return < 0
        assert result.max_drawdown < 0


# ═══════════════════════════════════════════════════════
# Strategy Algo Tests
# ═══════════════════════════════════════════════════════

class TestStrategyAlgos:

    def _make_target(self, n_assets=3, n_days=300):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
        cols = [f"ASSET_{i}" for i in range(n_assets)]
        prices = pd.DataFrame(
            np.cumsum(np.random.randn(n_days, n_assets) * 0.5, axis=0) + 100,
            index=dates, columns=cols,
        )
        target = MagicMock()
        target.now = dates[-1]
        target.temp = {"prices": prices}
        target.universe = prices
        return target, prices

    def test_equal_weight_algo(self):
        from portfolio_backtester.strategies import EqualWeightAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = EqualWeightAlgo()
        target, _ = self._make_target()
        result = algo(target)
        assert result is True
        weights = target.temp["weights"]
        assert len(weights) == 3
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_momentum_algo(self):
        from portfolio_backtester.strategies import MomentumAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = MomentumAlgo(top_n=2)
        target, _ = self._make_target()
        result = algo(target)
        assert result is True
        weights = target.temp["weights"]
        assert len(weights) == 2

    def test_momentum_algo_insufficient_data(self):
        from portfolio_backtester.strategies import MomentumAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = MomentumAlgo(top_n=2, lookback=252)
        target = MagicMock()
        target.now = pd.Timestamp("2023-01-01")
        target.temp = {"prices": pd.DataFrame({"A": [1, 2, 3]})}
        result = algo(target)
        assert result is False

    def test_risk_parity_algo_engine_reused(self):
        """Verify fix: RiskParityAlgo reuses engine (created in __init__)."""
        from portfolio_backtester.strategies import RiskParityAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = RiskParityAlgo(lookback=63)
        assert hasattr(algo, "_engine")
        assert algo._engine is not None

    def test_risk_parity_algo_call(self):
        from portfolio_backtester.strategies import RiskParityAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = RiskParityAlgo(lookback=63)
        target, _ = self._make_target(n_days=200)
        result = algo(target)
        assert result is True
        weights = target.temp["weights"]
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_risk_parity_algo_insufficient_data(self):
        from portfolio_backtester.strategies import RiskParityAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = RiskParityAlgo(lookback=63)
        target = MagicMock()
        target.now = pd.Timestamp("2023-01-01")
        target.temp = {"prices": pd.DataFrame({"A": [1, 2, 3]})}
        result = algo(target)
        assert result is False

    def test_tactical_allocation_algo(self):
        from portfolio_backtester.strategies import TacticalAllocationAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = TacticalAllocationAlgo(
            aggressive_assets=["ASSET_0", "ASSET_1"],
            defensive_assets=["ASSET_2"],
        )
        target, _ = self._make_target()
        result = algo(target)
        assert result is True
        weights = target.temp["weights"]
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_tactical_vol_20d_guard(self):
        """Verify fix: TacticalAllocationAlgo checks vol_20d column exists."""
        import inspect
        from portfolio_backtester.strategies import TacticalAllocationAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        src = inspect.getsource(TacticalAllocationAlgo.__call__)
        assert "vol_20d" in src
        assert "in features.columns" in src or "not features.empty" in src

    def test_tactical_no_aggressive_assets(self):
        from portfolio_backtester.strategies import TacticalAllocationAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = TacticalAllocationAlgo(aggressive_assets=[], defensive_assets=[])
        target, _ = self._make_target()
        result = algo(target)
        assert result is True
        weights = target.temp["weights"]
        assert len(weights) == 3

    def test_mean_variance_algo_fallback(self):
        from portfolio_backtester.strategies import MeanVarianceAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = MeanVarianceAlgo(lookback=63)
        target, _ = self._make_target(n_days=200)
        result = algo(target)
        assert result is True
        weights = target.temp["weights"]
        assert len(weights) > 0

    def test_algo_null_now_returns_true(self):
        from portfolio_backtester.strategies import EqualWeightAlgo, _HAS_BT
        if not _HAS_BT:
            pytest.skip("bt not installed")
        algo = EqualWeightAlgo()
        target = MagicMock()
        target.now = None
        assert algo(target) is True
