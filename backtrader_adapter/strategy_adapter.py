"""
Backtrader Strategy Adapter
=============================

Bridges stocks_plugin strategies into Backtrader's event-driven engine.

Usage:
    config = BacktraderConfig(commission=0.001)
    # Use with runner.run_backtest()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import backtrader as bt  # type: ignore[import-untyped]
    _HAS_BT = True
except ImportError:
    _HAS_BT = False


@dataclass
class BacktraderConfig:
    """Configuration for Backtrader backtests.

    Attributes:
        initial_capital: Starting portfolio value
        commission: Commission per trade (fraction, e.g., 0.001 = 0.1%)
        slippage_perc: Slippage as percentage (e.g., 0.001 = 0.1%)
        slippage_fixed: Fixed slippage in price units (overrides perc if set)
        use_bracket_orders: Enable bracket orders (stop-loss + take-profit)
        stop_loss_pct: Stop-loss percentage for bracket orders
        take_profit_pct: Take-profit percentage for bracket orders
        size_pct: Position size as fraction of portfolio (default 95%)
    """
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage_perc: float = 0.001
    slippage_fixed: Optional[float] = None
    use_bracket_orders: bool = False
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    size_pct: float = 0.95


if _HAS_BT:

    class StocksPluginBTStrategy(bt.Strategy):
        """Backtrader Strategy that delegates signal generation to a user function.

        The user provides a strategy_fn(context) -> Dict[str, int] where signals
        are -1 (sell/short), 0 (hold), +1 (buy/cover).

        Params:
            strategy_fn: Callable that receives a dict context and returns signals
            config: BacktraderConfig instance
            indicators: Dict of pre-computed indicator DataFrames
        """

        params = (
            ("strategy_fn", None),
            ("config", BacktraderConfig()),
            ("indicators", {}),
        )

        def __init__(self):
            self.order_refs = {}
            self.trade_log: List[Dict[str, Any]] = []
            self._bar_count = 0

            # Pre-compute indicators if TechnicalIndicators available
            try:
                from shared.indicators.technical_indicators import TechnicalIndicators as TI
                self._ti = TI
            except ImportError:
                self._ti = None

        def next(self):
            self._bar_count += 1
            if self.p.strategy_fn is None:
                return

            # Build context similar to BacktestContext
            context = {
                "bar_index": self._bar_count,
                "datetime": self.datas[0].datetime.datetime(0),
                "open": self.datas[0].open[0],
                "high": self.datas[0].high[0],
                "low": self.datas[0].low[0],
                "close": self.datas[0].close[0],
                "volume": self.datas[0].volume[0],
                "position_size": self.position.size,
                "portfolio_value": self.broker.getvalue(),
                "capital": self.broker.getcash(),
                "indicators": self.p.indicators,
            }

            signals = self.p.strategy_fn(context)
            if not isinstance(signals, dict):
                signals = {"default": signals}

            for name, signal in signals.items():
                if signal == 0:
                    continue

                data = self.datas[0]
                config = self.p.config

                if signal > 0 and self.position.size <= 0:
                    if self.position.size < 0:
                        self.close(data=data)

                    size = self._calc_size(data)
                    if size <= 0:
                        continue

                    if config.use_bracket_orders:
                        price = data.close[0]
                        stop_price = price * (1 - config.stop_loss_pct)
                        limit_price = price * (1 + config.take_profit_pct)
                        self.buy_bracket(
                            data=data, size=size,
                            stopprice=stop_price, limitprice=limit_price,
                        )
                    else:
                        self.buy(data=data, size=size)

                elif signal < 0 and self.position.size >= 0:
                    if self.position.size > 0:
                        self.close(data=data)

                    size = self._calc_size(data)
                    if size <= 0:
                        continue

                    if config.use_bracket_orders:
                        price = data.close[0]
                        stop_price = price * (1 + config.stop_loss_pct)
                        limit_price = price * (1 - config.take_profit_pct)
                        self.sell_bracket(
                            data=data, size=size,
                            stopprice=stop_price, limitprice=limit_price,
                        )
                    else:
                        self.sell(data=data, size=size)

        def _calc_size(self, data) -> int:
            """Calculate position size based on config."""
            config = self.p.config
            available = self.broker.getcash() * config.size_pct
            price = data.close[0]
            if price <= 0:
                return 0
            return int(available / price)

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trade_log.append({
                    "pnl": trade.pnl,
                    "pnlcomm": trade.pnlcomm,
                    "size": trade.size,
                    "price": trade.price,
                    "barlen": trade.barlen,
                })

else:
    StocksPluginBTStrategy = None  # type: ignore[assignment,misc]
