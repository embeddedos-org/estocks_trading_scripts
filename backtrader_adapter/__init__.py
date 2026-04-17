"""Backtrader integration adapter for stocks_plugin."""

try:
    from backtrader_adapter.runner import run_backtest
    from backtrader_adapter.strategy_adapter import BacktraderConfig, StocksPluginBTStrategy
    __all__ = ["run_backtest", "BacktraderConfig", "StocksPluginBTStrategy"]
except ImportError:
    __all__ = []
