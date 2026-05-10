"""
Backtrader Runner
==================

Convenience wrapper to run a backtest with Backtrader's Cerebro engine.

Usage:
    from backtrader_adapter.runner import run_backtest
    from backtrader_adapter.strategy_adapter import BacktraderConfig

    def my_strategy(ctx):
        if ctx["close"] > ctx.get("sma_20", 0):
            return {"default": 1}
        return {"default": -1}

    result = run_backtest(my_strategy, df, BacktraderConfig())
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import backtrader as bt  # type: ignore[import-untyped]
    _HAS_BT = True
except ImportError:
    _HAS_BT = False

from backtrader_adapter.strategy_adapter import BacktraderConfig

if _HAS_BT:
    from backtrader_adapter.data_feed import DataFrameFeed
    from backtrader_adapter.strategy_adapter import StocksPluginBTStrategy
    from backtrader_adapter.analyzers import BacktestResultAnalyzer, to_backtest_result_v2


def run_backtest(
    strategy_fn: Callable[[Dict[str, Any]], Dict[str, int]],
    data: pd.DataFrame,
    config: Optional[BacktraderConfig] = None,
    indicators: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run a backtest using Backtrader's Cerebro engine.

    Args:
        strategy_fn: Function that receives context dict, returns signal dict
            {symbol: -1/0/+1}
        data: OHLCV DataFrame with DatetimeIndex
        config: BacktraderConfig with commission, slippage, etc.
        indicators: Optional pre-computed indicators dict

    Returns:
        BacktestResultV2 with unified metrics

    Raises:
        ImportError: If backtrader is not installed
    """
    if not _HAS_BT:
        raise ImportError(
            "backtrader is required. Install: pip install backtrader"
        )

    if config is None:
        config = BacktraderConfig()
    if indicators is None:
        indicators = {}

    # Ensure DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if "date" in data.columns:
            data = data.set_index("date")
        data.index = pd.to_datetime(data.index)

    cerebro = bt.Cerebro()

    # Add data feed
    feed = DataFrameFeed(dataname=data)
    cerebro.adddata(feed)

    # Configure broker
    cerebro.broker.setcash(config.initial_capital)
    cerebro.broker.setcommission(commission=config.commission)

    if config.slippage_fixed is not None:
        cerebro.broker.set_slippage_fixed(config.slippage_fixed)
    else:
        cerebro.broker.set_slippage_perc(config.slippage_perc)

    # Add strategy
    cerebro.addstrategy(
        StocksPluginBTStrategy,
        strategy_fn=strategy_fn,
        config=config,
        indicators=indicators,
    )

    # Add analyzer
    cerebro.addanalyzer(BacktestResultAnalyzer)

    # Run
    logger.info("Running Backtrader backtest (capital=%.0f, bars=%d)", config.initial_capital, len(data))
    results = cerebro.run()

    # Convert to BacktestResultV2
    return to_backtest_result_v2(results[0], initial_capital=config.initial_capital)
