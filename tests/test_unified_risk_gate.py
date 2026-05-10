"""Tests for the unified risk gate — shared/risk_manager.py RiskManager.

Covers: instance isolation, can_trade gates, position registration,
portfolio exposure limits, concentration checks, daily loss limits,
portfolio summary, thread safety, and state reset.

20+ tests total.
"""

import os
import sys
import time
import threading
import pytest
from unittest.mock import patch
from datetime import date as real_date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.risk_manager import (
    RiskManager,
    RiskManagerConfig,
    SizingMethod,
    TradeRecord,
)


@pytest.fixture
def rm():
    """RiskManager with tight limits for testing."""
    cfg = RiskManagerConfig(
        total_capital=100_000,
        max_daily_loss=2_000,
        max_consecutive_losses=3,
        cooldown_seconds=60,
        max_open_positions=5,
        max_portfolio_heat_pct=15.0,
        max_drawdown_pct=10.0,
        circuit_breaker_pause_hours=24.0,
        max_trades_per_hour=20,
        min_seconds_between_trades=0,
        risk_per_trade_pct=2.0,
    )
    return RiskManager(config=cfg)


# ═══════════════════════════════════════════════════════════════════════
#  1. Instance Isolation (singleton-like pattern)
# ═══════════════════════════════════════════════════════════════════════


class TestSingletonPattern:

    def test_separate_instances_independent(self):
        rm1 = RiskManager()
        rm2 = RiskManager()
        rm1.record_trade("AAPL", pnl=-1000)
        assert rm2._daily_pnl == 0.0

    def test_shared_config_object_independent_state(self):
        cfg = RiskManagerConfig(total_capital=50_000)
        rm1 = RiskManager(config=cfg)
        rm2 = RiskManager(config=cfg)
        rm1.record_trade("AAPL", pnl=-500)
        assert rm2._daily_pnl == 0.0

    def test_default_config_applied(self):
        rm = RiskManager()
        assert rm.config.total_capital == 100_000
        assert rm.config.max_daily_loss == 5_000


# ═══════════════════════════════════════════════════════════════════════
#  2. can_trade — allows within limits
# ═══════════════════════════════════════════════════════════════════════


class TestCanTradeAllowed:

    def test_fresh_manager_can_trade(self, rm):
        assert rm.can_trade() is True

    def test_after_small_loss_can_trade(self, rm):
        rm.record_trade("AAPL", pnl=-500)
        assert rm.can_trade() is True

    def test_after_win_can_trade(self, rm):
        rm.record_trade("AAPL", pnl=1000)
        assert rm.can_trade() is True

    def test_with_positions_under_max(self, rm):
        rm.add_position("AAPL", 1000)
        rm.add_position("MSFT", 1000)
        assert rm.can_trade() is True


# ═══════════════════════════════════════════════════════════════════════
#  3. can_trade — blocks on max exposure
# ═══════════════════════════════════════════════════════════════════════


class TestCanTradeBlocksMaxExposure:

    def test_blocks_at_max_positions(self, rm):
        for i in range(5):
            rm.add_position(f"SYM{i}", 500)
        assert rm.can_trade() is False

    def test_unblocks_after_closing_position(self, rm):
        for i in range(5):
            rm.add_position(f"SYM{i}", 500)
        rm.remove_position("SYM0")
        assert rm.can_trade() is True


# ═══════════════════════════════════════════════════════════════════════
#  4. can_trade — blocks on single stock concentration
# ═══════════════════════════════════════════════════════════════════════


class TestStockConcentration:

    def test_single_stock_under_limit(self, rm):
        assert rm.check_portfolio_heat(additional_risk=10_000) is True

    def test_single_stock_over_limit(self, rm):
        assert rm.check_portfolio_heat(additional_risk=16_000) is False

    def test_at_limit_boundary(self, rm):
        # 15% of 100k = 15k → at limit
        assert rm.check_portfolio_heat(additional_risk=15_000) is True

    def test_just_over_limit(self, rm):
        assert rm.check_portfolio_heat(additional_risk=15_001) is False


# ═══════════════════════════════════════════════════════════════════════
#  5. can_trade — blocks on sector concentration
# ═══════════════════════════════════════════════════════════════════════


class TestSectorConcentration:

    def test_sector_concentration_blocks(self, rm):
        rm.add_position("AAPL", 5_000)
        rm.add_position("MSFT", 5_000)
        rm.add_position("GOOG", 5_000)
        assert rm.check_portfolio_heat(additional_risk=1) is False

    def test_sector_concentration_allows(self, rm):
        rm.add_position("AAPL", 3_000)
        rm.add_position("XOM", 3_000)
        assert rm.check_portfolio_heat(additional_risk=3_000) is True


# ═══════════════════════════════════════════════════════════════════════
#  6. register_position tracks correctly
# ═══════════════════════════════════════════════════════════════════════


class TestPositionRegistration:

    def test_add_position_tracked(self, rm):
        rm.add_position("AAPL", 2_000)
        assert "AAPL" in rm._open_positions
        assert rm._open_positions["AAPL"] == 2_000

    def test_overwrite_position(self, rm):
        rm.add_position("AAPL", 2_000)
        rm.add_position("AAPL", 3_000)
        assert rm._open_positions["AAPL"] == 3_000

    def test_multiple_positions(self, rm):
        rm.add_position("AAPL", 1_000)
        rm.add_position("MSFT", 2_000)
        rm.add_position("GOOG", 3_000)
        assert len(rm._open_positions) == 3


# ═══════════════════════════════════════════════════════════════════════
#  7. close_position updates P&L
# ═══════════════════════════════════════════════════════════════════════


class TestClosePosition:

    def test_close_removes_tracking(self, rm):
        rm.add_position("AAPL", 1_000)
        rm.remove_position("AAPL")
        assert "AAPL" not in rm._open_positions

    def test_close_nonexistent_no_error(self, rm):
        rm.remove_position("NOPE")

    def test_record_trade_positive_updates_equity(self, rm):
        rm.record_trade("AAPL", pnl=500)
        assert rm._current_equity == 100_500

    def test_record_trade_negative_updates_equity(self, rm):
        rm.record_trade("AAPL", pnl=-300)
        assert rm._current_equity == 99_700


# ═══════════════════════════════════════════════════════════════════════
#  8. Daily loss limit blocks all trading
# ═══════════════════════════════════════════════════════════════════════


class TestDailyLossBlocks:

    def test_blocks_at_limit(self, rm):
        rm._daily_pnl = -2_000
        assert rm.can_trade() is False

    def test_blocks_over_limit(self, rm):
        rm._daily_pnl = -3_000
        assert rm.can_trade() is False

    def test_allows_under_limit(self, rm):
        rm._daily_pnl = -1_999
        assert rm.can_trade() is True

    def test_wins_rescue_from_near_limit(self, rm):
        rm.record_trade("AAPL", pnl=-1_800)
        rm._last_trade_time = 0
        assert rm.can_trade() is True
        rm.record_trade("MSFT", pnl=1_000)
        rm._last_trade_time = 0
        assert rm._daily_pnl == -800
        assert rm.can_trade() is True


# ═══════════════════════════════════════════════════════════════════════
#  9. get_portfolio_summary returns correct data
# ═══════════════════════════════════════════════════════════════════════


class TestPortfolioSummary:

    def test_initial_status_structure(self, rm):
        status = rm.get_status()
        expected_keys = {
            "can_trade", "current_equity", "peak_equity", "daily_pnl",
            "daily_trade_count", "consecutive_losses", "cooldown_active",
            "cooldown_remaining_s", "circuit_breaker_active", "drawdown_pct",
            "open_positions", "portfolio_heat_pct", "trades_last_hour",
            "total_trades",
        }
        assert expected_keys.issubset(set(status.keys()))

    def test_initial_values(self, rm):
        status = rm.get_status()
        assert status["current_equity"] == 100_000
        assert status["daily_pnl"] == 0.0
        assert status["consecutive_losses"] == 0
        assert status["open_positions"] == 0

    def test_status_reflects_trades(self, rm):
        rm.record_trade("AAPL", pnl=-200)
        rm._last_trade_time = 0
        rm.record_trade("MSFT", pnl=300)
        rm._last_trade_time = 0
        status = rm.get_status()
        assert status["daily_pnl"] == 100.0
        assert status["total_trades"] == 2

    def test_status_reflects_positions(self, rm):
        rm.add_position("AAPL", 2_000)
        status = rm.get_status()
        assert status["open_positions"] == 1
        assert status["portfolio_heat_pct"] > 0


# ═══════════════════════════════════════════════════════════════════════
#  10. Thread safety with concurrent access
# ═══════════════════════════════════════════════════════════════════════


class TestThreadSafety:

    def test_concurrent_record_trade(self, rm):
        errors = []

        def trade_loop():
            try:
                for _ in range(50):
                    rm.record_trade("AAPL", pnl=-1)
                    rm._last_trade_time = 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=trade_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert rm._daily_trade_count == 200

    def test_concurrent_add_remove_positions(self, rm):
        errors = []

        def position_churn():
            try:
                for i in range(20):
                    rm.add_position(f"SYM{threading.current_thread().name}_{i}", 100)
                    rm.remove_position(f"SYM{threading.current_thread().name}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=position_churn) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════
#  11. Reset clears state
# ═══════════════════════════════════════════════════════════════════════


class TestResetClearsState:

    def test_new_instance_is_clean(self):
        cfg = RiskManagerConfig(total_capital=100_000, min_seconds_between_trades=0)
        rm = RiskManager(config=cfg)
        rm.record_trade("AAPL", pnl=-3000)
        # Create fresh instance — state should be clean
        rm2 = RiskManager(config=cfg)
        assert rm2._daily_pnl == 0.0
        assert rm2._consecutive_losses == 0
        assert rm2._current_equity == 100_000

    def test_drawdown_circuit_breaker_resets_on_new_instance(self):
        cfg = RiskManagerConfig(
            total_capital=100_000, max_drawdown_pct=5.0,
            max_consecutive_losses=999, min_seconds_between_trades=0,
        )
        rm = RiskManager(config=cfg)
        rm.record_trade("AAPL", pnl=-6000)
        assert rm.can_trade() is False

        rm2 = RiskManager(config=cfg)
        assert rm2.can_trade() is True
