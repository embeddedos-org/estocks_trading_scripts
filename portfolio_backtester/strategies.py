"""
Portfolio Backtesting Strategies (bt Algos)
============================================

Pre-built portfolio allocation strategies using the bt library.

Usage:
    import bt
    from portfolio_backtester.strategies import MomentumAlgo, RiskParityAlgo

    strategy = bt.Strategy("momentum", [bt.algos.RunMonthly(), MomentumAlgo(top_n=5)])
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import bt as bt_lib  # type: ignore[import-untyped]
    _HAS_BT = True
except ImportError:
    _HAS_BT = False
    logger.warning("bt not installed. Install: pip install bt")


if _HAS_BT:

    class MomentumAlgo(bt_lib.Algo):
        """Select top N assets by 12-1 month momentum (skip most recent month).

        Args:
            top_n: Number of assets to select
            lookback: Momentum lookback in trading days (default 252)
            skip: Recent days to skip (default 21, i.e., 1 month)
        """

        def __init__(self, top_n: int = 5, lookback: int = 252, skip: int = 21):
            super().__init__()
            self.top_n = top_n
            self.lookback = lookback
            self.skip = skip

        def __call__(self, target):
            if target.now is None:
                return True

            prices = target.temp.get("prices", target.universe)
            if prices is None or len(prices) < self.lookback:
                return False

            # 12-1 month momentum: return over lookback minus recent skip
            end_idx = len(prices) - self.skip
            if end_idx <= 0:
                return False
            start_idx = max(0, end_idx - self.lookback)

            momentum = prices.iloc[end_idx] / prices.iloc[start_idx] - 1
            momentum = momentum.dropna()

            if len(momentum) == 0:
                return False

            top = momentum.nlargest(min(self.top_n, len(momentum)))
            selected = list(top.index)

            weights = pd.Series(1.0 / len(selected), index=selected)
            target.temp["weights"] = weights
            return True

    class RiskParityAlgo(bt_lib.Algo):
        """Inverse-volatility weighting via RiskParityEngine.

        Args:
            lookback: Volatility estimation window (trading days)
        """

        def __init__(self, lookback: int = 63):
            super().__init__()
            self.lookback = lookback
            from portfolio_backtester.risk_parity import RiskParityEngine, RiskParityConfig
            self._engine = RiskParityEngine(RiskParityConfig(lookback=self.lookback))

        def __call__(self, target):
            if target.now is None:
                return True

            prices = target.temp.get("prices", target.universe)
            if prices is None or len(prices) < self.lookback + 1:
                return False

            returns = prices.pct_change().dropna()
            if len(returns) < self.lookback:
                return False

            weights = self._engine.inverse_vol_weights(returns)

            target.temp["weights"] = weights
            return True

    class TacticalAllocationAlgo(bt_lib.Algo):
        """Regime-based tactical allocation using MLRegimeClassifier.

        Shifts allocation between aggressive and defensive based on
        detected market regime.
        """

        def __init__(
            self,
            aggressive_assets: Optional[list] = None,
            defensive_assets: Optional[list] = None,
        ):
            super().__init__()
            self.aggressive = aggressive_assets or []
            self.defensive = defensive_assets or []

        def __call__(self, target):
            if target.now is None:
                return True

            prices = target.temp.get("prices", target.universe)
            if prices is None or len(prices) < 60:
                return False

            # Try to use MLRegimeClassifier for regime detection
            regime = "TRENDING"
            try:
                from shared.ml.regime_classifier import MLRegimeClassifier
                clf = MLRegimeClassifier()
                ref_col = prices.columns[0]
                ref_prices = prices[ref_col]
                dummy_df = pd.DataFrame({
                    "open": ref_prices, "high": ref_prices * 1.01,
                    "low": ref_prices * 0.99, "close": ref_prices,
                    "volume": 1_000_000,
                }, index=prices.index)
                features = clf.compute_features(dummy_df)
                if not features.empty and "vol_20d" in features.columns:
                    vol = features["vol_20d"].iloc[-1]
                    if vol > features["vol_20d"].quantile(0.9):
                        regime = "VOLATILE"
            except Exception:
                pass

            all_assets = list(prices.columns)
            if regime == "VOLATILE" and self.defensive:
                selected = [a for a in self.defensive if a in all_assets]
            else:
                selected = [a for a in self.aggressive if a in all_assets] or all_assets

            if not selected:
                selected = all_assets

            weights = pd.Series(1.0 / len(selected), index=selected)
            target.temp["weights"] = weights
            return True

    class MeanVarianceAlgo(bt_lib.Algo):
        """Mean-variance optimal weights (delegates to mean_variance.py).

        Falls back to equal-weight if mean_variance module unavailable.
        """

        def __init__(self, lookback: int = 126):
            super().__init__()
            self.lookback = lookback

        def __call__(self, target):
            if target.now is None:
                return True

            prices = target.temp.get("prices", target.universe)
            if prices is None or len(prices) < self.lookback + 1:
                return False

            returns = prices.pct_change().dropna()
            if len(returns) < self.lookback:
                return False

            recent = returns.tail(self.lookback)

            try:
                from shared.strategies.mean_variance import MeanVarianceOptimizer
                opt = MeanVarianceOptimizer(recent)
                weights = opt.max_sharpe()
            except ImportError:
                weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)

            target.temp["weights"] = weights
            return True

    class EqualWeightAlgo(bt_lib.Algo):
        """Simple 1/N equal-weight allocation."""

        def __call__(self, target):
            if target.now is None:
                return True

            prices = target.temp.get("prices", target.universe)
            if prices is None:
                return False

            n = len(prices.columns)
            weights = pd.Series(1.0 / n, index=prices.columns)
            target.temp["weights"] = weights
            return True

else:
    MomentumAlgo = None  # type: ignore[assignment,misc]
    RiskParityAlgo = None  # type: ignore[assignment,misc]
    TacticalAllocationAlgo = None  # type: ignore[assignment,misc]
    MeanVarianceAlgo = None  # type: ignore[assignment,misc]
    EqualWeightAlgo = None  # type: ignore[assignment,misc]
