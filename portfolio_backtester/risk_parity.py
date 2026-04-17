"""
Risk Parity Engine
====================

Inverse-volatility, target-volatility, and risk-budgeting weight schemes.

Usage:
    from portfolio_backtester.risk_parity import RiskParityEngine, RiskParityConfig
    engine = RiskParityEngine(RiskParityConfig(target_vol=0.10))
    weights = engine.inverse_vol_weights(returns_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskParityConfig:
    """Configuration for risk parity calculations.

    Attributes:
        target_vol: Annualized target volatility (e.g., 0.10 = 10%)
        lookback: Rolling window for volatility estimation (trading days)
        rebalance_freq: Rebalance frequency in trading days
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
    """
    target_vol: float = 0.10
    lookback: int = 63
    rebalance_freq: int = 21
    min_weight: float = 0.0
    max_weight: float = 1.0


class RiskParityEngine:
    """Computes portfolio weights using risk parity approaches."""

    def __init__(self, config: Optional[RiskParityConfig] = None):
        self.config = config or RiskParityConfig()

    def inverse_vol_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Compute inverse-volatility weights.

        Each asset's weight is proportional to 1/volatility.

        Args:
            returns: DataFrame of asset returns (columns = assets)

        Returns:
            Series of weights summing to 1.0
        """
        vol = returns.tail(self.config.lookback).std() * np.sqrt(252)
        inv_vol = 1.0 / vol.replace(0, np.nan)
        inv_vol = inv_vol.fillna(0)
        total = inv_vol.sum()
        if total == 0:
            return pd.Series(1.0 / len(returns.columns), index=returns.columns)
        weights = inv_vol / total
        return self._clip_weights(weights)

    def target_vol_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Compute weights to achieve target portfolio volatility.

        Scales inverse-vol weights so the portfolio's expected annualized
        volatility matches config.target_vol.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Series of weights (may sum to < 1.0 if leverage not allowed)
        """
        base_weights = self.inverse_vol_weights(returns)
        recent = returns.tail(self.config.lookback)
        cov = recent.cov() * 252
        w = base_weights.values
        port_vol = float(np.sqrt(w @ cov.values @ w))
        if port_vol == 0:
            return base_weights
        scale = self.config.target_vol / port_vol
        scaled = base_weights * min(scale, 1.0)
        return self._clip_weights(scaled)

    def risk_budgeting(
        self, returns: pd.DataFrame, budgets: Optional[pd.Series] = None
    ) -> pd.Series:
        """Compute risk-budgeting weights.

        Each asset contributes a specified fraction of total portfolio risk.
        Default budget is equal risk contribution (ERC).

        Args:
            returns: DataFrame of asset returns
            budgets: Series of risk budgets per asset (must sum to 1.0).
                If None, uses equal budgets.

        Returns:
            Series of weights
        """
        n = len(returns.columns)
        if budgets is None:
            budgets = pd.Series(1.0 / n, index=returns.columns)

        recent = returns.tail(self.config.lookback)
        cov = recent.cov().values * 252

        # Iterative solver: start from inverse-vol, then adjust
        w = np.ones(n) / n
        for _ in range(100):
            sigma_w = cov @ w
            rc = w * sigma_w
            total_rc = rc.sum()
            if total_rc == 0:
                break
            rc_pct = rc / total_rc
            # Adjust weights toward target budgets
            adjustment = budgets.values / np.where(rc_pct > 0, rc_pct, 1e-10)
            w = w * adjustment
            w = np.maximum(w, 1e-10)
            w = w / w.sum()

        weights = pd.Series(w, index=returns.columns)
        return self._clip_weights(weights)

    def _clip_weights(self, weights: pd.Series) -> pd.Series:
        """Clip weights to min/max bounds and renormalize."""
        clipped = weights.clip(lower=self.config.min_weight, upper=self.config.max_weight)
        total = clipped.sum()
        if total > 0:
            clipped = clipped / total
        return clipped
