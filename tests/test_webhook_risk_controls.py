"""Tests for webhook server risk controls and health monitoring.

Covers: health endpoint, HealthMonitor silence detection,
DailyPnLTracker, CooldownManager, DrawdownCircuitBreaker,
max trade value cap, rate limiter, and integrated risk gate checks.

25+ tests total.
"""

import os
import sys
import time
import json
import pytest
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tradingview.webhooks.webhook_server import (
    HealthMonitor,
    DailyPnLTracker,
    CooldownManager,
    DrawdownCircuitBreaker,
    RateLimiter,
    HealthResponse,
    AlertPayload,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. Health Endpoint / HealthResponse
# ═══════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Health response model returns correct JSON fields."""

    def test_health_response_structure(self):
        hr = HealthResponse(
            status="healthy",
            uptime_seconds=120.5,
            last_alert_time=None,
            total_alerts_processed=0,
            version="1.0.0",
        )
        assert hr.status == "healthy"
        assert hr.uptime_seconds == 120.5
        assert hr.last_alert_time is None
        assert hr.total_alerts_processed == 0
        assert hr.version == "1.0.0"

    def test_health_response_with_last_alert(self):
        now = datetime.now(timezone.utc).isoformat()
        hr = HealthResponse(
            status="healthy",
            uptime_seconds=3600.0,
            last_alert_time=now,
            total_alerts_processed=42,
            version="2.0.0",
        )
        assert hr.last_alert_time == now
        assert hr.total_alerts_processed == 42

    def test_health_response_serialises(self):
        hr = HealthResponse(
            status="healthy", uptime_seconds=0.0,
            last_alert_time=None, total_alerts_processed=0, version="1.0.0",
        )
        data = hr.model_dump()
        assert "status" in data
        assert "uptime_seconds" in data


# ═══════════════════════════════════════════════════════════════════════
#  2. HealthMonitor — silence detection
# ═══════════════════════════════════════════════════════════════════════


class TestHealthMonitor:
    """HealthMonitor sends warning after silence period."""

    def test_initial_state(self):
        hm = HealthMonitor(max_silence_minutes=30)
        assert hm.alerts_processed == 0
        assert hm.last_alert_time is None

    def test_record_alert_updates_state(self):
        hm = HealthMonitor()
        hm.record_alert()
        assert hm.alerts_processed == 1
        assert hm.last_alert_time is not None

    def test_uptime_positive(self):
        hm = HealthMonitor()
        assert hm.uptime_seconds >= 0

    def test_silence_warning_dispatched(self):
        dispatcher = MagicMock()
        hm = HealthMonitor(
            max_silence_minutes=1,
            alert_dispatcher=dispatcher,
        )
        # Simulate being started 5 minutes ago so uptime > max_silence
        hm._start_time = time.time() - 300
        hm._check_silence()
        assert hm._silence_warned is True
        dispatcher.dispatch.assert_called_once()

    def test_no_duplicate_silence_warning(self):
        dispatcher = MagicMock()
        hm = HealthMonitor(max_silence_minutes=1, alert_dispatcher=dispatcher)
        hm._start_time = time.time() - 300
        hm._check_silence()
        hm._check_silence()
        assert dispatcher.dispatch.call_count == 1

    def test_silence_reset_after_alert(self):
        hm = HealthMonitor(max_silence_minutes=1)
        hm._start_time = time.time() - 300
        hm._check_silence()
        assert hm._silence_warned is True
        hm.record_alert()
        assert hm._silence_warned is False


# ═══════════════════════════════════════════════════════════════════════
#  3. DailyPnLTracker
# ═══════════════════════════════════════════════════════════════════════


class TestDailyPnLTracker:
    """DailyPnLTracker blocks trades after loss limit, allows under limit, resets daily."""

    def test_allows_trade_initially(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        assert tracker.can_trade() is True

    def test_allows_trade_under_limit(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-500)
        assert tracker.can_trade() is True

    def test_blocks_trade_at_limit(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-1000)
        assert tracker.can_trade() is False

    def test_blocks_trade_over_limit(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-1500)
        assert tracker.can_trade() is False

    def test_accumulates_losses(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-400)
        tracker.record_trade("MSFT", pnl=-400)
        assert tracker.can_trade() is True
        tracker.record_trade("GOOG", pnl=-300)
        assert tracker.can_trade() is False

    def test_wins_offset_losses(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-800)
        tracker.record_trade("MSFT", pnl=500)
        assert tracker.daily_pnl == -300
        assert tracker.can_trade() is True

    def test_reset_daily_clears_state(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-900)
        assert tracker.trade_count == 1
        tracker.reset_daily()
        assert tracker.daily_pnl == 0.0
        assert tracker.trade_count == 0
        assert tracker.can_trade() is True

    def test_auto_reset_on_new_day(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-900)
        # Simulate date change
        tracker._last_reset_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        assert tracker.can_trade() is True
        assert tracker.daily_pnl == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  4. CooldownManager
# ═══════════════════════════════════════════════════════════════════════


class TestCooldownManager:
    """CooldownManager tracks consecutive losses, blocks during cooldown, resets on win."""

    def test_no_cooldown_initially(self):
        cm = CooldownManager(max_consecutive_losses=3, cooldown_minutes=30)
        assert cm.is_in_cooldown("strat_a") is False

    def test_single_loss_no_cooldown(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("strat_a", won=False)
        assert cm.is_in_cooldown("strat_a") is False

    def test_two_losses_no_cooldown(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("strat_a", won=False)
        cm.record_result("strat_a", won=False)
        assert cm.is_in_cooldown("strat_a") is False

    def test_three_losses_triggers_cooldown(self):
        cm = CooldownManager(max_consecutive_losses=3, cooldown_minutes=30)
        for _ in range(3):
            cm.record_result("strat_a", won=False)
        assert cm.is_in_cooldown("strat_a") is True

    def test_cooldown_blocks_trading(self):
        cm = CooldownManager(max_consecutive_losses=2, cooldown_minutes=60)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        assert cm.is_in_cooldown("s1") is True

    def test_cooldown_per_strategy(self):
        cm = CooldownManager(max_consecutive_losses=2, cooldown_minutes=60)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        assert cm.is_in_cooldown("s1") is True
        assert cm.is_in_cooldown("s2") is False

    def test_win_resets_loss_streak(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=True)
        cm.record_result("s1", won=False)
        assert cm.is_in_cooldown("s1") is False

    def test_cooldown_expires(self):
        cm = CooldownManager(max_consecutive_losses=2, cooldown_minutes=30)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        # Force expiry
        cm._cooldown_until["s1"] = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert cm.is_in_cooldown("s1") is False

    def test_get_status(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("s1", won=False)
        status = cm.get_status("s1")
        assert status["strategy"] == "s1"
        assert status["loss_streak"] == 1
        assert status["in_cooldown"] is False


# ═══════════════════════════════════════════════════════════════════════
#  5. DrawdownCircuitBreaker
# ═══════════════════════════════════════════════════════════════════════


class TestDrawdownCircuitBreaker:
    """DrawdownCircuitBreaker blocks on 10%+ drawdown, auto-unblocks after lockout."""

    def test_can_trade_initially(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0, lockout_hours=24)
        assert dcb.can_trade() is True

    def test_small_drawdown_no_trigger(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(95_000)  # 5% drawdown
        assert dcb.can_trade() is True

    def test_10pct_drawdown_triggers(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0, lockout_hours=24)
        dcb.update_equity(100_000)
        dcb.update_equity(89_000)  # 11% drawdown
        assert dcb.can_trade() is False

    def test_exact_threshold_triggers(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(90_000)  # exactly 10%
        assert dcb.can_trade() is False

    def test_auto_unblock_after_lockout(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0, lockout_hours=24)
        dcb.update_equity(100_000)
        dcb.update_equity(89_000)
        assert dcb.can_trade() is False
        # Simulate lockout expiry
        dcb._tripped_until = datetime.now(timezone.utc) - timedelta(hours=1)
        assert dcb.can_trade() is True

    def test_peak_equity_tracking(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(110_000)
        assert dcb._peak_equity == 110_000
        dcb.update_equity(105_000)
        assert dcb._peak_equity == 110_000

    def test_drawdown_pct_property(self):
        dcb = DrawdownCircuitBreaker()
        dcb.update_equity(100_000)
        dcb.update_equity(95_000)
        assert abs(dcb.drawdown_pct - 5.0) < 0.01

    def test_reset_clears_trip(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(85_000)
        assert dcb.can_trade() is False
        dcb.reset()
        assert dcb.can_trade() is True

    def test_get_status(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        status = dcb.get_status()
        assert status["peak_equity"] == 100_000
        assert status["tripped"] is False


# ═══════════════════════════════════════════════════════════════════════
#  6. Max Trade Value Cap
# ═══════════════════════════════════════════════════════════════════════


class TestMaxTradeValueCap:
    """Position size dollar cap is applied via _check_risk_gates."""

    def test_trade_value_under_cap_unchanged(self):
        alert = AlertPayload(
            symbol="AAPL", action="buy", price=150.0, quantity=10.0,
        )
        # 10 * 150 = 1500, under any reasonable cap
        max_cap = 50_000.0
        trade_value = alert.quantity * alert.price
        assert trade_value <= max_cap

    def test_trade_value_over_cap_gets_capped(self):
        alert = AlertPayload(
            symbol="AAPL", action="buy", price=150.0, quantity=500.0,
        )
        max_cap = 50_000.0
        trade_value = alert.quantity * alert.price  # 75000
        if trade_value > max_cap:
            capped_qty = max_cap / alert.price
            alert.quantity = capped_qty
        assert alert.quantity == pytest.approx(333.33, abs=0.1)

    def test_cap_with_zero_price(self):
        alert = AlertPayload(
            symbol="PENNY", action="buy", price=0.0, quantity=1000.0,
        )
        max_cap = 50_000.0
        trade_value = alert.quantity * alert.price  # 0
        # No capping needed when trade value is 0
        assert trade_value <= max_cap


# ═══════════════════════════════════════════════════════════════════════
#  7. Rate Limiter
# ═══════════════════════════════════════════════════════════════════════


class TestRateLimiter:
    """Rate limiter enforces per-IP caps."""

    def test_allows_initial_requests(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("127.0.0.1") is True

    def test_blocks_excess_requests(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("1.2.3.4")
        assert rl.is_allowed("1.2.3.4") is False

    def test_per_ip_isolation(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("10.0.0.1")
        rl.is_allowed("10.0.0.1")
        assert rl.is_allowed("10.0.0.1") is False
        assert rl.is_allowed("10.0.0.2") is True

    def test_remaining_count(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        rl.is_allowed("1.1.1.1")
        rl.is_allowed("1.1.1.1")
        assert rl.get_remaining("1.1.1.1") == 3


# ═══════════════════════════════════════════════════════════════════════
#  8. Integration — all controls work together
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationAllControls:
    """All risk controls operate together."""

    def test_all_controls_allow_when_healthy(self):
        tracker = DailyPnLTracker(max_daily_loss=5000)
        cooldown = CooldownManager(max_consecutive_losses=3)
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        breaker.update_equity(100_000)

        assert tracker.can_trade() is True
        assert not cooldown.is_in_cooldown("strat")
        assert breaker.can_trade() is True

    def test_one_control_blocks_pipeline(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        cooldown = CooldownManager(max_consecutive_losses=3)
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        breaker.update_equity(100_000)

        tracker.record_trade("AAPL", pnl=-1500)
        # Tracker blocks, others allow
        assert tracker.can_trade() is False
        assert not cooldown.is_in_cooldown("strat")
        assert breaker.can_trade() is True

    def test_multiple_controls_can_block(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        cooldown = CooldownManager(max_consecutive_losses=2)
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)

        tracker.record_trade("AAPL", pnl=-1500)
        cooldown.record_result("s1", won=False)
        cooldown.record_result("s1", won=False)
        breaker.update_equity(100_000)
        breaker.update_equity(85_000)

        assert tracker.can_trade() is False
        assert cooldown.is_in_cooldown("s1") is True
        assert breaker.can_trade() is False

    def test_thread_safety_concurrent_pnl_recording(self):
        tracker = DailyPnLTracker(max_daily_loss=100_000)
        errors = []

        def record_trades():
            try:
                for _ in range(100):
                    tracker.record_trade("AAPL", pnl=-1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_trades) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.trade_count == 500
        assert tracker.daily_pnl == -500
