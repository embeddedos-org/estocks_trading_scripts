"""
Mean-Variance Optimization
=============================

Classical Markowitz, Black-Litterman, and Risk Budgeting optimization.

Usage:
    from shared.strategies.mean_variance import MeanVarianceOptimizer, BlackLitterman
    opt = MeanVarianceOptimizer(returns_df)
    weights = opt.max_sharpe()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    long_only: bool = True
    target_return: Optional[float] = None
    max_positions: Optional[int] = None


class MeanVarianceOptimizer:
    """Classical Markowitz mean-variance optimization."""

    def __init__(self, returns: pd.DataFrame, rf: float = 0.0):
        """
        Args:
            returns: DataFrame of asset returns (columns = tickers)
            rf: Risk-free rate (annualized)
        """
        if not _HAS_SCIPY:
            raise ImportError("scipy required. Install: pip install scipy")
        self.returns = returns
        self.rf = rf
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        self.n_assets = len(returns.columns)
        self.tickers = list(returns.columns)

    def max_sharpe(self, constraints: Optional[OptimizationConstraints] = None) -> pd.Series:
        """Find the maximum Sharpe ratio portfolio."""
        c = constraints or OptimizationConstraints()

        def neg_sharpe(w):
            port_ret = np.dot(w, self.mean_returns)
            port_vol = np.sqrt(w @ self.cov_matrix.values @ w)
            if port_vol == 0:
                return 0
            return -(port_ret - self.rf) / port_vol

        bounds = [(c.min_weight, c.max_weight)] * self.n_assets
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if c.target_return is not None:
            cons.append({"type": "eq", "fun": lambda w, _c=c: np.dot(w, self.mean_returns) - _c.target_return})

        w0 = np.ones(self.n_assets) / self.n_assets
        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)

        weights = pd.Series(result.x, index=self.tickers)
        if c.max_positions:
            # Keep only top N positions
            top = weights.abs().nlargest(c.max_positions).index
            weights[~weights.index.isin(top)] = 0
            weights = weights / weights.sum()

        return weights

    def min_variance(self, constraints: Optional[OptimizationConstraints] = None) -> pd.Series:
        """Find the minimum variance portfolio."""
        c = constraints or OptimizationConstraints()

        def portfolio_var(w):
            return w @ self.cov_matrix.values @ w

        bounds = [(c.min_weight, c.max_weight)] * self.n_assets
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        w0 = np.ones(self.n_assets) / self.n_assets
        result = minimize(portfolio_var, w0, method="SLSQP", bounds=bounds, constraints=cons)
        return pd.Series(result.x, index=self.tickers)

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Compute the efficient frontier.

        Returns:
            DataFrame with return, volatility, sharpe for each point
        """
        min_ret = float(self.mean_returns.min())
        max_ret = float(self.mean_returns.max())
        target_returns = np.linspace(min_ret, max_ret, n_points)

        results = []
        for target in target_returns:
            c = OptimizationConstraints(target_return=target)
            try:
                weights = self.min_variance(c)
                port_ret = np.dot(weights, self.mean_returns)
                port_vol = np.sqrt(weights.values @ self.cov_matrix.values @ weights.values)
                sharpe = (port_ret - self.rf) / port_vol if port_vol > 0 else 0
                results.append({"return": port_ret, "volatility": port_vol, "sharpe": sharpe})
            except Exception:
                continue

        return pd.DataFrame(results)


class BlackLitterman:
    """Black-Litterman model for combining market equilibrium with investor views."""

    def __init__(self, returns: pd.DataFrame, market_caps: Optional[pd.Series] = None, rf: float = 0.0):
        self.returns = returns
        self.cov = returns.cov() * 252
        self.tickers = list(returns.columns)
        self.rf = rf
        self._views = []
        self._view_confidences = []

        # Market-cap weights (or equal weight)
        if market_caps is not None:
            self.market_weights = market_caps / market_caps.sum()
        else:
            self.market_weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)

        # Risk aversion (implied from market)
        port_var = self.market_weights.values @ self.cov.values @ self.market_weights.values
        market_ret = float(returns.mean().dot(self.market_weights) * 252)
        self.risk_aversion = (market_ret - rf) / port_var if port_var > 0 else 2.5

        # Equilibrium returns
        self.equilibrium_returns = self.risk_aversion * self.cov @ self.market_weights

    def add_view(self, asset: str, expected_return: float, confidence: float = 0.5):
        """Add an absolute view on an asset's expected return.

        Args:
            asset: Ticker symbol
            expected_return: Expected annualized return
            confidence: Confidence in view (0 to 1)
        """
        if asset not in self.tickers:
            raise ValueError(f"Unknown asset: {asset}")
        self._views.append((asset, expected_return))
        self._view_confidences.append(confidence)

    def posterior_returns(self) -> pd.Series:
        """Compute posterior expected returns combining equilibrium and views."""
        if not self._views:
            return self.equilibrium_returns

        n = len(self.tickers)
        k = len(self._views)

        # P matrix (pick matrix)
        P = np.zeros((k, n))
        q = np.zeros(k)
        for i, (asset, ret) in enumerate(self._views):
            j = self.tickers.index(asset)
            P[i, j] = 1.0
            q[i] = ret

        # Omega (uncertainty of views)
        tau = 1.0 / len(self.returns)
        omega_diag = []
        for i in range(k):
            conf = self._view_confidences[i]
            # Higher confidence -> lower uncertainty
            omega_diag.append((1 - conf) / conf * (P[i] @ (tau * self.cov.values) @ P[i]))
        Omega = np.diag(omega_diag)

        # Posterior
        Sigma = self.cov.values
        tau_Sigma = tau * Sigma
        inv_tau_Sigma = np.linalg.inv(tau_Sigma)
        inv_Omega = np.linalg.inv(Omega)

        posterior_cov = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P)
        posterior_mean = posterior_cov @ (inv_tau_Sigma @ self.equilibrium_returns.values + P.T @ inv_Omega @ q)

        return pd.Series(posterior_mean, index=self.tickers)

    def optimal_weights(self, constraints: Optional[OptimizationConstraints] = None) -> pd.Series:
        """Compute optimal weights from posterior returns."""
        posterior = self.posterior_returns()
        # Use MeanVarianceOptimizer with posterior returns
        opt = MeanVarianceOptimizer(self.returns, self.rf)
        opt.mean_returns = posterior
        return opt.max_sharpe(constraints)


class RiskBudgeting:
    """Risk budgeting optimization.

    Allocates each asset a specified risk budget so that its
    marginal risk contribution matches the budget.
    """

    def __init__(self, returns: pd.DataFrame):
        if not _HAS_SCIPY:
            raise ImportError("scipy required. Install: pip install scipy")
        self.returns = returns
        self.cov = returns.cov().values * 252
        self.tickers = list(returns.columns)
        self.n = len(self.tickers)

    def optimize(self, budgets: Optional[pd.Series] = None) -> pd.Series:
        """Compute risk-budgeting optimal weights.

        Args:
            budgets: Risk budget per asset (sums to 1). Default: equal.

        Returns:
            Series of optimal weights
        """
        if budgets is None:
            b = np.ones(self.n) / self.n
        else:
            b = budgets.values

        # Try cvxpy first, fallback to scipy
        if _HAS_CVXPY:
            return self._optimize_cvxpy(b)
        return self._optimize_scipy(b)

    def _optimize_scipy(self, budgets: np.ndarray) -> pd.Series:
        """Scipy-based risk budgeting."""
        def objective(w):
            sigma_w = self.cov @ w
            rc = w * sigma_w
            total_risk = np.sqrt(w @ self.cov @ w)
            rc_pct = rc / (total_risk ** 2) if total_risk > 0 else rc
            return np.sum((rc_pct - budgets) ** 2)

        bounds = [(1e-6, 1.0)] * self.n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(self.n) / self.n

        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
        return pd.Series(result.x, index=self.tickers)

    def _optimize_cvxpy(self, budgets: np.ndarray) -> pd.Series:
        """CVXPY-based risk budgeting using log-barrier."""
        w = cp.Variable(self.n)
        risk = cp.quad_form(w, self.cov)
        log_barrier = cp.sum(budgets @ cp.log(w))

        prob = cp.Problem(
            cp.Minimize(risk - log_barrier),
            [cp.sum(w) == 1, w >= 1e-6],
        )
        prob.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            logger.warning("CVXPY solve failed, falling back to scipy")
            return self._optimize_scipy(budgets)

        weights = w.value / w.value.sum()
        return pd.Series(weights, index=self.tickers)
