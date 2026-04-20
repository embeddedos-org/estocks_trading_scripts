"""Backtrader integration adapter for stocks_plugin."""

import logging

_logger = logging.getLogger(__name__)

try:
    from backtrader_adapter.runner import run_backtest
    from backtrader_adapter.strategy_adapter import BacktraderConfig, StocksPluginBTStrategy
    __all__ = ["run_backtest", "BacktraderConfig", "StocksPluginBTStrategy"]
except ImportError as e:
    _logger.warning("backtrader_adapter imports failed: %s", e)
    __all__ = []
