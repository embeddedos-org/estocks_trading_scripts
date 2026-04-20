"""
Factor Models and Alpha Ranking
==================================

Fama-French style factor computation, alpha ranking, and long-short backtesting.

Usage:
    from shared.strategies.factor_models import FamaFrenchFactors, AlphaRanker
    ff = FamaFrenchFactors()
    factors = ff.compute_factors(universe_prices, market_caps)
    ranker = AlphaRanker()
    ranked = ranker.rank(universe_prices, factors)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FamaFrenchFactors:
    """Compute Fama-French style factors from a stock universe."""

    def compute_factors(
        self,
        prices: pd.DataFrame,
        market_caps: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute SMB, HML, MOM, Quality, and LowVol factors.

        Args:
            prices: DataFrame of prices (columns = tickers, index = dates)
            market_caps: Optional DataFrame of market caps (same shape)

        Returns:
            DataFrame with factor columns
        """
        returns = prices.pct_change()
        factors = pd.DataFrame(index=prices.index)

        # SMB (Small Minus Big) - size factor
        if market_caps is not None:
            median_cap = market_caps.median(axis=1)
            small = market_caps.lt(median_cap, axis=0)
            big = ~small
            factors["SMB"] = returns.where(small).mean(axis=1) - returns.where(big).mean(axis=1)
        else:
            # Proxy: use price as size proxy
            median_price = prices.median(axis=1)
            small = prices.lt(median_price, axis=0)
            factors["SMB"] = returns.where(small).mean(axis=1) - returns.where(~small).mean(axis=1)

        # HML (High Minus Low) - value factor using price-to-52w-high ratio
        high_52w = prices.rolling(252).max()
        value_ratio = prices / high_52w.replace(0, np.nan)
        median_val = value_ratio.median(axis=1)
        high_val = value_ratio.lt(median_val, axis=0)  # Low price/52w = value
        factors["HML"] = returns.where(high_val).mean(axis=1) - returns.where(~high_val).mean(axis=1)

        # MOM (Momentum) - 12-1 month momentum
        mom_12_1 = prices.shift(21) / prices.shift(252) - 1
        median_mom = mom_12_1.median(axis=1)
        winners = mom_12_1.gt(median_mom, axis=0)
        factors["MOM"] = returns.where(winners).mean(axis=1) - returns.where(~winners).mean(axis=1)

        # Quality - rolling Sharpe ratio as quality proxy
        rolling_mean = returns.rolling(252).mean()
        rolling_std = returns.rolling(252).std()
        quality = rolling_mean / rolling_std.replace(0, np.nan)
        factors["Quality"] = quality.mean(axis=1)

        # LowVol factor
        vol_60 = returns.rolling(60).std()
        median_vol = vol_60.median(axis=1)
        low_vol = vol_60.lt(median_vol, axis=0)
        factors["LowVol"] = returns.where(low_vol).mean(axis=1) - returns.where(~low_vol).mean(axis=1)

        return factors.dropna()


class AlphaRanker:
    """Rank stocks by composite factor z-scores."""

    def rank(
        self,
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Rank universe by weighted factor composite.

        Args:
            prices: Price DataFrame (columns = tickers)
            factors: Factor returns DataFrame from FamaFrenchFactors
            weights: Dict of factor -> weight. Default: equal weight.

        Returns:
            DataFrame with composite z-scores per stock
        """
        returns = prices.pct_change()
        n_stocks = len(prices.columns)

        if weights is None:
            weights = {col: 1.0 / len(factors.columns) for col in factors.columns}

        # Compute factor exposures per stock
        exposures = pd.DataFrame(index=prices.columns)

        # Momentum exposure
        mom = prices.iloc[-21] / prices.iloc[-252] - 1 if len(prices) > 252 else prices.pct_change(63).iloc[-1]
        exposures["MOM"] = (mom - mom.mean()) / mom.std() if mom.std() > 0 else 0

        # Volatility exposure (low = good)
        vol = returns.tail(60).std()
        exposures["LowVol"] = -(vol - vol.mean()) / vol.std() if vol.std() > 0 else 0

        # Value exposure
        high_52w = prices.rolling(252).max().iloc[-1] if len(prices) > 252 else prices.max()
        val = prices.iloc[-1] / high_52w.replace(0, np.nan)
        exposures["HML"] = -(val - val.mean()) / val.std() if val.std() > 0 else 0

        # Composite
        composite = pd.Series(0.0, index=prices.columns)
        for factor, weight in weights.items():
            if factor in exposures.columns:
                composite += exposures[factor] * weight

        result = pd.DataFrame({"composite_zscore": composite})
        result = result.sort_values("composite_zscore", ascending=False)
        result["rank"] = range(1, len(result) + 1)
        return result


class FactorBacktester:
    """Backtest a long-short factor portfolio."""

    def backtest_long_short(
        self,
        prices: pd.DataFrame,
        factor_name: str = "MOM",
        n_long: int = 5,
        n_short: int = 5,
        rebalance_freq: int = 21,
    ) -> Any:
        """Run long-short factor backtest.

        Args:
            prices: Universe price DataFrame
            factor_name: Which factor to use for ranking
            n_long: Number of long positions
            n_short: Number of short positions
            rebalance_freq: Rebalance every N bars

        Returns:
            BacktestResultV2
        """
        from shared.backtesting.backtest_engine_v2 import BacktestEngineV2

        returns = prices.pct_change()
        tickers = list(prices.columns)

        def strategy_fn(ctx):
            signals = {}
            idx = ctx.bar_index
            if idx % rebalance_freq != 0 or idx < 252:
                return {t: 0 for t in tickers}

            # Rank by selected factor
            current_prices = {}
            for t in tickers:
                if t in ctx.bars and len(ctx.bars[t]) > 252:
                    p = ctx.bars[t]["close"]
                    if factor_name == "MOM":
                        score = p.iloc[-21] / p.iloc[-252] - 1
                    elif factor_name == "LowVol":
                        score = -p.pct_change().tail(60).std()
                    elif factor_name == "HML":
                        high_52w = p.rolling(252).max().iloc[-1]
                        score = -(p.iloc[-1] / high_52w) if high_52w > 0 else 0
                    elif factor_name == "Quality":
                        ret = p.pct_change()
                        std = ret.tail(252).std()
                        score = ret.tail(252).mean() / std if std > 0 else 0
                    else:
                        score = p.iloc[-21] / p.iloc[-252] - 1
                    current_prices[t] = score

            if len(current_prices) < n_long + n_short:
                return {t: 0 for t in tickers}

            sorted_tickers = sorted(current_prices, key=current_prices.get, reverse=True)
            for t in tickers:
                if t in sorted_tickers[:n_long]:
                    signals[t] = 1
                elif t in sorted_tickers[-n_short:]:
                    signals[t] = -1
                else:
                    signals[t] = 0

            return signals

        engine = BacktestEngineV2()
        data = {t: prices[[t]].rename(columns={t: "close"}) for t in tickers}
        # Add dummy OHLV columns
        for t in data:
            df = data[t]
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = 1_000_000

        engine.load_data(data)
        return engine.run(strategy_fn)
