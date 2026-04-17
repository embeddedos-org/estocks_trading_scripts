"""Advanced quantitative trading strategies."""

__all__ = []

try:
    from shared.strategies.factor_models import FamaFrenchFactors, AlphaRanker, FactorBacktester
    __all__.extend(["FamaFrenchFactors", "AlphaRanker", "FactorBacktester"])
except ImportError:
    pass

try:
    from shared.strategies.stat_arb import CointegrationScanner, OrnsteinUhlenbeck, BasketTrader
    __all__.extend(["CointegrationScanner", "OrnsteinUhlenbeck", "BasketTrader"])
except ImportError:
    pass

try:
    from shared.strategies.mean_variance import MeanVarianceOptimizer, BlackLitterman, RiskBudgeting
    __all__.extend(["MeanVarianceOptimizer", "BlackLitterman", "RiskBudgeting"])
except ImportError:
    pass
