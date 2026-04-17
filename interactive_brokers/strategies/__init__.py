"""Interactive Brokers pre-built trading strategies."""

from interactive_brokers.strategies.pairs_trading import PairsTradingBot
from interactive_brokers.strategies.options_wheel import OptionsWheelStrategy

__all__ = [
    "PairsTradingBot",
    "OptionsWheelStrategy",
]
