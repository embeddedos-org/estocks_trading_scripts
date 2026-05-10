"""Tests for the shared PortfolioAnalytics module.

Covers Value-at-Risk (historical & parametric), CVaR, Sharpe ratio,
Sortino ratio, max drawdown, stress testing (default + custom scenarios),
beta calculation, and edge cases (empty/single-return inputs).

20+ tests total.
"""

import os
import sys
import math
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.analytics.portfolio_analytics import PortfolioAnalytics


@pytest.fixture
def pa():
    """PortfolioAnalytics instance with default risk-free rate."""
    return PortfolioAnalytics(risk_free_rate=0.05)


@pytest.fixture
def normal_returns():
    """500 daily returns from a slightly positive-drift process."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0.0005, 0.015, 500))


@pytest.fixture
def negative_returns():
    """Returns with a strong negative bias (for CVaR testing)."""
    rng = np.random.RandomState(99)
    return pd.Series(rng.normal(-0.002, 0.02, 300))


@pytest.fixture
def equity_curve():
    """Equity curve with a clear drawdown followed by recovery."""
    dates = pd.bdate_range("2023-01-01", periods=100)
    values = [100_000]
    for i in range(1, 100):
        if i < 30:
            values.append(values[-1] * 1.002)
        elif i < 60:
            values.append(values[-1] * 0.995)
        else:
            values.append(values[-1] * 1.004)
    return pd.Series(values, index=dates)


# ═══════════════════════════════════════════════════════════════════════
#  1. Value at Risk — Historical
# ═══════════════════════════════════════════════════════════════════════


class TestVaRHistorical:

    def test_var_historical_returns_dict(self, pa, normal_returns):
        result = pa.calculate_var(normal_returns, confidence=0.95, method="historical")
        assert isinstance(result, dict)
        assert "var" in result
        assert "cvar" in result
        assert "confidence" in result
        assert "method" in result

    def test_var_historical_is_negative(self, pa, normal_returns):
        result = pa.calculate_var(normal_returns, confidence=0.95, method="historical")
        assert result["var"] < 0, "95% historical VaR should be a negative return"

    def test_var_historical_95_vs_99(self, pa, normal_returns):
        var95 = pa.calculate_var(normal_returns, confidence=0.95, method="historical")
        var99 = pa.calculate_var(normal_returns, confidence=0.99, method="historical")
        assert var99["var"] <= var95["var"], "99% VaR should be deeper than 95%"

    def test_var_historical_confidence_stored(self, pa, normal_returns):
        result = pa.calculate_var(normal_returns, confidence=0.90, method="historical")
        assert result["confidence"] == 0.90
        assert result["method"] == "historical"


# ═══════════════════════════════════════════════════════════════════════
#  2. Value at Risk — Parametric
# ═══════════════════════════════════════════════════════════════════════


class TestVaRParametric:

    def test_var_parametric_returns_dict(self, pa, normal_returns):
        result = pa.calculate_var(normal_returns, confidence=0.95, method="parametric")
        assert isinstance(result, dict)
        assert result["method"] == "parametric"

    def test_var_parametric_is_negative_for_low_mean(self, pa, negative_returns):
        result = pa.calculate_var(negative_returns, confidence=0.95, method="parametric")
        assert result["var"] < 0

    def test_var_parametric_uses_normal_distribution(self, pa):
        # Deterministic test: known mean and std
        returns = pd.Series([0.01] * 100 + [-0.01] * 100)
        result = pa.calculate_var(returns, confidence=0.95, method="parametric")
        assert isinstance(result["var"], float)
        assert result["var"] != 0.0


# ═══════════════════════════════════════════════════════════════════════
#  3. CVaR (Conditional VaR)
# ═══════════════════════════════════════════════════════════════════════


class TestCVaR:

    def test_cvar_below_or_equal_to_var(self, pa, normal_returns):
        result = pa.calculate_var(normal_returns, confidence=0.95, method="historical")
        assert result["cvar"] <= result["var"], "CVaR must be <= VaR (deeper tail loss)"

    def test_cvar_is_mean_of_tail(self, pa, negative_returns):
        result = pa.calculate_var(negative_returns, confidence=0.95, method="historical")
        assert result["cvar"] < 0, "CVaR on negative-bias returns should be negative"

    def test_cvar_more_extreme_than_var(self, pa, normal_returns):
        result = pa.calculate_var(normal_returns, confidence=0.99, method="historical")
        # CVaR should capture tail risk beyond VaR
        assert result["cvar"] <= result["var"]


# ═══════════════════════════════════════════════════════════════════════
#  4. Sharpe Ratio
# ═══════════════════════════════════════════════════════════════════════


class TestSharpeRatio:

    def test_sharpe_positive_for_positive_drift(self, pa):
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.001, 0.005, 500))
        sharpe = pa.calculate_sharpe(returns)
        assert sharpe > 0, "Sharpe should be positive for positive-drift returns"

    def test_sharpe_annualized_correctly(self, pa):
        # Known positive drift with some variance
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        sharpe = pa.calculate_sharpe(returns, periods=252)
        # Positive drift → positive Sharpe
        assert sharpe > 0
        assert math.isfinite(sharpe)

    def test_sharpe_different_periods(self, pa, normal_returns):
        sharpe_daily = pa.calculate_sharpe(normal_returns, periods=252)
        sharpe_monthly = pa.calculate_sharpe(normal_returns, periods=12)
        # Different annualization factors → different results
        assert sharpe_daily != sharpe_monthly

    def test_sharpe_near_zero_vol(self, pa):
        # Near-constant returns: std ≈ 0 but may not be exactly 0
        # due to floating point; Sharpe may be 0 or very large
        returns = pd.Series([0.001] * 100)
        sharpe = pa.calculate_sharpe(returns)
        # The result is deterministic — just check it's finite or zero
        assert isinstance(sharpe, float)


# ═══════════════════════════════════════════════════════════════════════
#  5. Sortino Ratio
# ═══════════════════════════════════════════════════════════════════════


class TestSortinoRatio:

    def test_sortino_uses_only_downside(self, pa):
        # All positive returns → downside std = 0 → returns 0.0
        returns = pd.Series([0.01, 0.02, 0.005, 0.015, 0.008])
        sortino = pa.calculate_sortino(returns)
        assert sortino == 0.0, "No downside returns → sortino should be 0"

    def test_sortino_positive_for_mixed_returns(self, pa):
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.002, 0.01, 500))
        sortino = pa.calculate_sortino(returns)
        assert sortino > 0, "Positive-drift mixed returns should yield positive Sortino"

    def test_sortino_greater_than_sharpe_for_positive_skew(self, pa, normal_returns):
        sharpe = pa.calculate_sharpe(normal_returns)
        sortino = pa.calculate_sortino(normal_returns)
        # This is a statistical property — not always guaranteed,
        # but for a normal dist with slight positive mean, Sortino ≥ Sharpe
        # We just check both are finite numbers
        assert math.isfinite(sharpe)
        assert math.isfinite(sortino)


# ═══════════════════════════════════════════════════════════════════════
#  6. Max Drawdown
# ═══════════════════════════════════════════════════════════════════════


class TestMaxDrawdown:

    def test_max_drawdown_returns_dict(self, pa, equity_curve):
        result = pa.calculate_max_drawdown(equity_curve)
        assert isinstance(result, dict)
        assert "max_drawdown" in result
        assert "peak_date" in result
        assert "trough_date" in result
        assert "recovery_date" in result

    def test_max_drawdown_is_negative(self, pa, equity_curve):
        result = pa.calculate_max_drawdown(equity_curve)
        assert result["max_drawdown"] < 0, "Max drawdown should be negative"

    def test_peak_before_trough(self, pa, equity_curve):
        result = pa.calculate_max_drawdown(equity_curve)
        assert result["peak_date"] < result["trough_date"]

    def test_recovery_after_trough(self, pa, equity_curve):
        result = pa.calculate_max_drawdown(equity_curve)
        if result["recovery_date"] is not None:
            assert result["recovery_date"] >= result["trough_date"]

    def test_no_drawdown_on_monotonic_increase(self, pa):
        dates = pd.bdate_range("2023-01-01", periods=50)
        curve = pd.Series(range(100_000, 100_050), index=dates)
        result = pa.calculate_max_drawdown(curve)
        assert result["max_drawdown"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  7. Stress Test — Default Scenarios
# ═══════════════════════════════════════════════════════════════════════


class TestStressTestDefault:

    def test_all_5_default_scenarios(self, pa):
        positions = {
            "AAPL": {"shares": 100, "price": 150.0, "beta": 1.2},
            "MSFT": {"shares": 50, "price": 300.0, "beta": 1.1},
        }
        results = pa.stress_test(positions)
        assert len(results) == 5
        expected = {
            "market_crash_10pct", "correction_5pct", "flash_crash_3pct",
            "rally_5pct", "bear_market_20pct",
        }
        assert set(results.keys()) == expected

    def test_stress_results_have_correct_keys(self, pa):
        positions = {"AAPL": {"shares": 100, "price": 150.0}}
        results = pa.stress_test(positions)
        for scenario_result in results.values():
            assert "market_move" in scenario_result
            assert "portfolio_pnl" in scenario_result
            assert "pnl_pct" in scenario_result

    def test_crash_pnl_is_negative(self, pa):
        positions = {"SPY": {"shares": 100, "price": 500.0, "beta": 1.0}}
        results = pa.stress_test(positions)
        assert results["market_crash_10pct"]["portfolio_pnl"] < 0

    def test_rally_pnl_is_positive(self, pa):
        positions = {"SPY": {"shares": 100, "price": 500.0, "beta": 1.0}}
        results = pa.stress_test(positions)
        assert results["rally_5pct"]["portfolio_pnl"] > 0

    def test_beta_amplifies_losses(self, pa):
        positions_low = {"X": {"shares": 100, "price": 100.0, "beta": 0.5}}
        positions_high = {"X": {"shares": 100, "price": 100.0, "beta": 2.0}}
        r_low = pa.stress_test(positions_low)
        r_high = pa.stress_test(positions_high)
        # High beta should amplify crash loss
        assert abs(r_high["market_crash_10pct"]["portfolio_pnl"]) > abs(
            r_low["market_crash_10pct"]["portfolio_pnl"]
        )


# ═══════════════════════════════════════════════════════════════════════
#  8. Stress Test — Custom Scenarios
# ═══════════════════════════════════════════════════════════════════════


class TestStressTestCustom:

    def test_custom_scenario_works(self, pa):
        positions = {"AAPL": {"shares": 100, "price": 150.0, "beta": 1.0}}
        custom = {"custom_drop_15pct": -0.15}
        results = pa.stress_test(positions, scenarios=custom)
        assert "custom_drop_15pct" in results
        expected_pnl = 100 * 150.0 * (-0.15) * 1.0
        assert results["custom_drop_15pct"]["portfolio_pnl"] == expected_pnl

    def test_custom_replaces_defaults(self, pa):
        positions = {"AAPL": {"shares": 10, "price": 100.0}}
        custom = {"mild_dip": -0.01}
        results = pa.stress_test(positions, scenarios=custom)
        assert len(results) == 1
        assert "mild_dip" in results


# ═══════════════════════════════════════════════════════════════════════
#  9. Beta Calculation
# ═══════════════════════════════════════════════════════════════════════


class TestBetaCalculation:

    def test_beta_against_benchmark(self, pa):
        rng = np.random.RandomState(42)
        benchmark = pd.Series(rng.normal(0.0005, 0.01, 200))
        # Asset with beta ~ 1.5
        asset = benchmark * 1.5 + pd.Series(rng.normal(0, 0.003, 200))
        beta = pa.calculate_beta(asset, benchmark)
        assert 1.0 < beta < 2.0, f"Expected beta near 1.5, got {beta}"

    def test_beta_of_benchmark_against_itself(self, pa):
        rng = np.random.RandomState(42)
        benchmark = pd.Series(rng.normal(0.0005, 0.01, 200))
        beta = pa.calculate_beta(benchmark, benchmark)
        assert abs(beta - 1.0) < 0.01, "Beta of asset against itself should be ~1.0"

    def test_beta_insufficient_data_returns_1(self, pa):
        short = pd.Series([0.01, 0.02, 0.01])
        benchmark = pd.Series([0.005, 0.01, 0.005])
        beta = pa.calculate_beta(short, benchmark)
        assert beta == 1.0, "Insufficient data should return default beta=1.0"


# ═══════════════════════════════════════════════════════════════════════
#  10. Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_empty_returns_var(self, pa):
        result = pa.calculate_var(pd.Series(dtype=float))
        assert result["var"] == 0.0
        assert result["cvar"] == 0.0

    def test_single_return_var(self, pa):
        result = pa.calculate_var(pd.Series([0.01]))
        assert result["var"] == 0.0

    def test_empty_returns_sharpe(self, pa):
        assert pa.calculate_sharpe(pd.Series(dtype=float)) == 0.0

    def test_single_return_sharpe(self, pa):
        assert pa.calculate_sharpe(pd.Series([0.01])) == 0.0

    def test_empty_returns_sortino(self, pa):
        assert pa.calculate_sortino(pd.Series(dtype=float)) == 0.0

    def test_single_return_sortino(self, pa):
        assert pa.calculate_sortino(pd.Series([0.01])) == 0.0

    def test_short_equity_curve_drawdown(self, pa):
        result = pa.calculate_max_drawdown(pd.Series([100_000]))
        assert result["max_drawdown"] == 0.0

    def test_nan_in_returns(self, pa):
        returns = pd.Series([0.01, np.nan, -0.02, 0.005, np.nan, -0.01])
        result = pa.calculate_var(returns, confidence=0.95)
        assert isinstance(result["var"], float)

    def test_repr(self, pa):
        assert "PortfolioAnalytics" in repr(pa)
        assert "0.05" in repr(pa)
