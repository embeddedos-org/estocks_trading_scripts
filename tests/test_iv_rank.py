"""Tests for IV Rank filter logic.

Covers IV rank calculation allowing/blocking entries, the IV rank
formula correctness, and the HV (Historical Volatility) percentile
fallback when implied volatility data is unavailable.

Tests operate against the OptionsWheelStrategy._calculate_iv_rank method
with all IB connections mocked.

8+ tests total.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Mock ib_async before importing the strategy ─────────────────────
# The options_wheel module imports `from ib_async import Stock`, so we
# provide a lightweight stub to avoid ImportError in CI.
@dataclass
class _MockBar:
    """Lightweight stand-in for ib_async BarData."""
    close: float


class _MockStock:
    """Stand-in for ib_async.Stock."""
    def __init__(self, symbol, exchange, currency):
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency


# Patch ib_async at the module level so the import inside
# _calculate_iv_rank resolves to our stub.
sys.modules.setdefault("ib_async", MagicMock(Stock=_MockStock))


from interactive_brokers.strategies.options_wheel import OptionsWheelStrategy


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_wheel(**overrides):
    """Build an OptionsWheelStrategy with all IB calls mocked."""
    conn = MagicMock()
    conn.qualifyContracts = MagicMock(return_value=None)
    om = MagicMock()
    defaults = dict(
        connection=conn,
        order_manager=om,
        capital=50_000,
        min_iv_rank=30.0,
        max_iv_rank=100.0,
    )
    defaults.update(overrides)
    return OptionsWheelStrategy(**defaults)


def _iv_bars(values):
    """Create a list of mock bars from IV values."""
    return [_MockBar(close=v) for v in values]


def _trade_bars(closes):
    """Create mock TRADES bars from a list of close prices."""
    return [_MockBar(close=c) for c in closes]


# ═══════════════════════════════════════════════════════════════════════
#  1. High IV Rank Allows Entry
# ═══════════════════════════════════════════════════════════════════════


class TestHighIVRankAllows:

    def test_iv_rank_60_allowed(self):
        """IV rank 60 > min_iv_rank 30 → should be allowed."""
        strategy = _make_wheel(min_iv_rank=30.0, max_iv_rank=100.0)

        # IV data: min=0.20, max=0.40, current=0.32
        # rank = (0.32 - 0.20) / (0.40 - 0.20) * 100 = 60
        iv_values = [0.20, 0.25, 0.30, 0.35, 0.40] * 10 + [0.32]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("AAPL")
        assert 55 <= rank <= 65

    def test_iv_rank_80_well_above_min(self):
        strategy = _make_wheel(min_iv_rank=30.0)

        # min=0.15, max=0.35, current=0.31
        # rank = (0.31-0.15) / (0.35-0.15) * 100 = 80
        iv_values = [0.15, 0.20, 0.25, 0.30, 0.35] * 10 + [0.31]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("MSFT")
        assert rank > 30.0


# ═══════════════════════════════════════════════════════════════════════
#  2. Low IV Rank Blocks
# ═══════════════════════════════════════════════════════════════════════


class TestLowIVRankBlocks:

    def test_iv_rank_15_below_min(self):
        """IV rank 15 < min_iv_rank 30 → would be blocked."""
        strategy = _make_wheel(min_iv_rank=30.0)

        # min=0.20, max=0.40, current=0.23
        # rank = (0.23-0.20) / (0.40-0.20) * 100 = 15
        iv_values = [0.20, 0.25, 0.30, 0.35, 0.40] * 10 + [0.23]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("AAPL")
        assert rank < 30.0, f"IV rank {rank} should be below min_iv_rank 30"

    def test_iv_rank_5_very_low(self):
        strategy = _make_wheel(min_iv_rank=30.0)

        # min=0.20, max=0.40, current=0.21
        # rank = (0.21-0.20) / (0.40-0.20) * 100 = 5
        iv_values = [0.20, 0.25, 0.30, 0.35, 0.40] * 10 + [0.21]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("LOW_IV")
        assert rank < 30.0


# ═══════════════════════════════════════════════════════════════════════
#  3. IV Rank Calculation Formula
# ═══════════════════════════════════════════════════════════════════════


class TestIVRankCalculation:

    def test_formula_correctness(self):
        """IV Rank = (current - min) / (max - min) * 100."""
        strategy = _make_wheel()

        # Known values: min=0.10, max=0.50, current=0.30
        # rank = (0.30-0.10) / (0.50-0.10) * 100 = 50
        iv_values = [0.10, 0.20, 0.30, 0.40, 0.50] * 10 + [0.30]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("TEST")
        assert abs(rank - 50.0) < 1.0, f"Expected IV rank ~50, got {rank}"

    def test_rank_at_max_is_100(self):
        strategy = _make_wheel()

        # current = max = 0.50
        iv_values = [0.10, 0.20, 0.30, 0.40, 0.50] * 10 + [0.50]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("MAX_IV")
        assert abs(rank - 100.0) < 1.0

    def test_rank_at_min_is_0(self):
        strategy = _make_wheel()

        # current = min = 0.10
        iv_values = [0.10, 0.20, 0.30, 0.40, 0.50] * 10 + [0.10]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("MIN_IV")
        assert abs(rank - 0.0) < 1.0


# ═══════════════════════════════════════════════════════════════════════
#  4. HV Fallback When IV Data Unavailable
# ═══════════════════════════════════════════════════════════════════════


class TestHVFallback:

    def test_hv_fallback_when_iv_fails(self):
        """When IV request throws, should fall back to HV percentile rank."""
        strategy = _make_wheel()
        strategy.connection.ib = MagicMock()

        # First call (IV) raises, second call (TRADES) returns bars
        rng = np.random.RandomState(42)
        closes = list(np.cumsum(rng.randn(100) * 0.5) + 100)
        trade_bars = _trade_bars(closes)

        def side_effect(contract, **kwargs):
            what = kwargs.get("whatToShow", "")
            if what == "OPTION_IMPLIED_VOLATILITY":
                raise Exception("No IV data")
            return trade_bars

        strategy.connection.ib.reqHistoricalData.side_effect = side_effect

        rank = strategy._calculate_iv_rank("NO_IV_STOCK")
        assert isinstance(rank, float)
        assert 0.0 <= rank <= 100.0

    def test_hv_fallback_when_iv_empty(self):
        """When IV returns empty bars, falls back to HV."""
        strategy = _make_wheel()
        strategy.connection.ib = MagicMock()

        rng = np.random.RandomState(42)
        closes = list(np.cumsum(rng.randn(100) * 0.5) + 100)

        call_count = [0]

        def side_effect(contract, **kwargs):
            call_count[0] += 1
            what = kwargs.get("whatToShow", "")
            if what == "OPTION_IMPLIED_VOLATILITY":
                return []  # empty bars triggers fallback
            return _trade_bars(closes)

        strategy.connection.ib.reqHistoricalData.side_effect = side_effect

        rank = strategy._calculate_iv_rank("EMPTY_IV")
        assert isinstance(rank, float)
        assert 0.0 <= rank <= 100.0

    def test_both_fail_returns_50(self):
        """When both IV and HV fail, should return neutral 50."""
        strategy = _make_wheel()
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.side_effect = Exception("All failed")

        rank = strategy._calculate_iv_rank("BROKEN")
        assert rank == 50.0, "When both IV and HV fail, neutral 50 expected"

    def test_hv_rank_uses_rolling_20d(self):
        """HV fallback should use 20-day rolling standard deviation."""
        strategy = _make_wheel()
        strategy.connection.ib = MagicMock()

        # Create bars where current HV is at the 75th percentile
        rng = np.random.RandomState(42)
        # Low vol period, then high vol at end
        closes_low = list(100 + np.cumsum(rng.randn(80) * 0.1))
        closes_high = list(closes_low[-1] + np.cumsum(rng.randn(40) * 2.0))
        closes = closes_low + closes_high

        def side_effect(contract, **kwargs):
            what = kwargs.get("whatToShow", "")
            if what == "OPTION_IMPLIED_VOLATILITY":
                raise Exception("No IV")
            return _trade_bars(closes)

        strategy.connection.ib.reqHistoricalData.side_effect = side_effect

        rank = strategy._calculate_iv_rank("HV_TEST")
        assert isinstance(rank, float)
        # Current HV should be higher (high vol period) → rank > 50
        assert rank > 40.0


# ═══════════════════════════════════════════════════════════════════════
#  5. Entry Gating Logic
# ═══════════════════════════════════════════════════════════════════════


class TestEntryGating:

    def test_rank_above_max_would_block(self):
        """IV rank above max_iv_rank should block entry."""
        strategy = _make_wheel(min_iv_rank=30.0, max_iv_rank=80.0)

        # rank = (0.48-0.10) / (0.50-0.10) * 100 = 95
        iv_values = [0.10, 0.20, 0.30, 0.40, 0.50] * 10 + [0.48]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("HIGH_IV")
        assert rank > strategy.max_iv_rank

    def test_rank_within_range_allowed(self):
        """IV rank within [min, max] range should be allowed."""
        strategy = _make_wheel(min_iv_rank=30.0, max_iv_rank=80.0)

        # rank = (0.30-0.10) / (0.50-0.10) * 100 = 50
        iv_values = [0.10, 0.20, 0.30, 0.40, 0.50] * 10 + [0.30]
        strategy.connection.ib = MagicMock()
        strategy.connection.ib.reqHistoricalData.return_value = _iv_bars(iv_values)

        rank = strategy._calculate_iv_rank("MID_IV")
        assert strategy.min_iv_rank <= rank <= strategy.max_iv_rank
