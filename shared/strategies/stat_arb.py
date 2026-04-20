"""
Statistical Arbitrage
=======================

Cointegration scanning, Ornstein-Uhlenbeck parameter estimation,
and basket trading.

Usage:
    from shared.strategies.stat_arb import CointegrationScanner, OrnsteinUhlenbeck
    scanner = CointegrationScanner()
    pairs = scanner.scan(universe_prices)
    ou = OrnsteinUhlenbeck()
    ou.fit(spread_series)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import coint, adfuller
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    logger.warning("statsmodels not installed for cointegration tests")


class CointegrationScanner:
    """Scan a universe of assets for cointegrated pairs.

    Reuses the Engle-Granger cointegration test pattern from
    interactive_brokers/strategies/pairs_trading.py.
    """

    def scan(
        self,
        prices: pd.DataFrame,
        significance: float = 0.05,
        min_obs: int = 60,
    ) -> List[Dict[str, Any]]:
        """Scan all pairs for cointegration.

        Args:
            prices: DataFrame of prices (columns = tickers)
            significance: P-value threshold
            min_obs: Minimum observations required

        Returns:
            List of dicts with {pair, p_value, test_stat, hedge_ratio}
            sorted by p_value ascending
        """
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels required. Install: pip install statsmodels")

        tickers = list(prices.columns)
        results = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                a, b = tickers[i], tickers[j]
                pa = prices[a].dropna()
                pb = prices[b].dropna()

                # Align
                common = pa.index.intersection(pb.index)
                if len(common) < min_obs:
                    continue

                pa_aligned = pa.loc[common]
                pb_aligned = pb.loc[common]

                try:
                    score, pvalue, _ = coint(pa_aligned, pb_aligned)

                    if pvalue < significance:
                        # Compute hedge ratio via OLS
                        X = sm.add_constant(pb_aligned.values)
                        model = sm.OLS(pa_aligned.values, X).fit()
                        hedge_ratio = float(model.params[1])

                        results.append({
                            "pair": (a, b),
                            "p_value": float(pvalue),
                            "test_stat": float(score),
                            "hedge_ratio": hedge_ratio,
                            "r_squared": float(model.rsquared),
                            "n_obs": len(common),
                        })
                except Exception as e:
                    logger.debug("Coint test failed for %s/%s: %s", a, b, e)

        results.sort(key=lambda x: x["p_value"])
        logger.info("Found %d cointegrated pairs (p < %.3f) from %d possible",
                    len(results), significance, len(tickers) * (len(tickers) - 1) // 2)
        return results


class OrnsteinUhlenbeck:
    """Ornstein-Uhlenbeck process parameter estimation for mean-reverting spreads."""

    def __init__(self):
        self.theta = None  # Mean-reversion speed
        self.mu = None     # Long-run mean
        self.sigma = None  # Volatility
        self._half_life = None

    def fit(self, spread: pd.Series) -> Dict[str, float]:
        """Fit OU parameters to a spread series.

        Uses OLS regression: dS = theta * (mu - S) * dt + sigma * dW

        Args:
            spread: Time series of spread values

        Returns:
            Dict with theta, mu, sigma, half_life
        """
        s = spread.dropna().values
        if len(s) < 10:
            raise ValueError("Need at least 10 data points")

        ds = np.diff(s)
        s_lag = s[:-1]

        # OLS: ds = a + b * s_lag
        X = np.column_stack([np.ones(len(s_lag)), s_lag])
        beta = np.linalg.lstsq(X, ds, rcond=None)[0]

        a, b = beta[0], beta[1]

        self.theta = -b  # Mean-reversion speed (annualized: * 252)
        self.mu = -a / b if b != 0 else np.mean(s)
        residuals = ds - (a + b * s_lag)
        self.sigma = float(np.std(residuals))
        self._half_life = np.log(2) / self.theta if self.theta > 0 else float("inf")

        return {
            "theta": float(self.theta),
            "mu": float(self.mu),
            "sigma": self.sigma,
            "half_life": float(self._half_life),
        }

    def optimal_entry_exit(
        self, current_spread: float, transaction_cost: float = 0.0
    ) -> Dict[str, float]:
        """Compute optimal entry/exit z-scores based on OU parameters.

        Returns:
            Dict with entry_long, entry_short, exit_long, exit_short thresholds
        """
        if self.theta is None:
            raise RuntimeError("Call fit() first")

        # Entry at +/- 2 sigma, exit near mean
        std = self.sigma / np.sqrt(2 * self.theta) if self.theta > 0 else self.sigma
        cost_adj = transaction_cost / std if std > 0 else 0

        return {
            "entry_long": self.mu - 2 * std - cost_adj,
            "entry_short": self.mu + 2 * std + cost_adj,
            "exit_long": self.mu - 0.5 * std,
            "exit_short": self.mu + 0.5 * std,
            "ou_std": float(std),
        }

    def half_life(self) -> float:
        """Return estimated half-life of mean reversion in bars."""
        if self._half_life is None:
            raise RuntimeError("Call fit() first")
        return self._half_life


class BasketTrader:
    """Trade a spread (basket) of cointegrated assets."""

    def __init__(self, ticker_a: str, ticker_b: str, hedge_ratio: float):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.hedge_ratio = hedge_ratio

    def compute_spread(self, prices: pd.DataFrame) -> pd.Series:
        """Compute spread = price_a - hedge_ratio * price_b."""
        return (prices[self.ticker_a] - self.hedge_ratio * prices[self.ticker_b]).rename("spread")

    def generate_signals(
        self, prices: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 20,
    ) -> pd.DataFrame:
        """Generate z-score based trading signals.

        Returns:
            DataFrame with spread, zscore, signal columns.
            Signal: 1 = long spread, -1 = short spread, 0 = flat
        """
        spread = self.compute_spread(prices)
        roll_mean = spread.rolling(lookback).mean()
        roll_std = spread.rolling(lookback).std().replace(0, np.nan)
        zscore = (spread - roll_mean) / roll_std

        signal = pd.Series(0, index=prices.index)
        position = 0
        for i in range(len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue
            if position == 0:
                if z < -entry_z:
                    position = 1
                elif z > entry_z:
                    position = -1
            elif position == 1 and z > -exit_z:
                position = 0
            elif position == -1 and z < exit_z:
                position = 0
            signal.iloc[i] = position

        return pd.DataFrame({"spread": spread, "zscore": zscore, "signal": signal})

    def backtest(self, prices: pd.DataFrame, **kwargs) -> Any:
        """Run pairs trading backtest via BacktestEngineV2."""
        from shared.backtesting.backtest_engine_v2 import BacktestEngineV2

        signals_df = self.generate_signals(prices, **kwargs)

        def strategy_fn(ctx):
            idx = ctx.bar_index
            if idx >= len(signals_df):
                return {self.ticker_a: 0, self.ticker_b: 0}
            sig = signals_df["signal"].iloc[idx]
            return {
                self.ticker_a: int(sig),
                self.ticker_b: int(-sig),
            }

        data = {}
        for t in [self.ticker_a, self.ticker_b]:
            df = pd.DataFrame({"close": prices[t], "open": prices[t],
                             "high": prices[t], "low": prices[t],
                             "volume": 1_000_000}, index=prices.index)
            data[t] = df

        engine = BacktestEngineV2()
        engine.load_data(data)
        return engine.run(strategy_fn)
