"""Technical indicators package with TA-Lib acceleration."""

from shared.indicators.technical_indicators import TechnicalIndicators

__all__ = ["TechnicalIndicators"]

try:
    from shared.indicators.candlestick_patterns import CandlestickPatterns
    __all__.append("CandlestickPatterns")
except ImportError:
    pass
