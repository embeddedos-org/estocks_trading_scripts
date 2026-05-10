"""Tests for the MarketHours utility module.

Covers market-open detection, weekend/holiday checks, premarket and
afterhours session logic, end-of-day flatten signals, time-to-close
calculation, and holiday detection.

12+ tests total.
"""

import os
import sys
import pytest
from datetime import datetime, date, time, timedelta, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.utils.market_hours import MarketHours


def _utc(year, month, day, hour, minute=0):
    """Build a timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


@pytest.fixture
def mh():
    """MarketHours with default settings (EST offset -5, no extended hours)."""
    return MarketHours(allow_premarket=False, allow_afterhours=False, timezone_offset=-5)


@pytest.fixture
def mh_extended():
    """MarketHours with premarket and afterhours enabled."""
    return MarketHours(allow_premarket=True, allow_afterhours=True, timezone_offset=-5)


# ═══════════════════════════════════════════════════════════════════════
#  1. Market Open During Regular Hours
# ═══════════════════════════════════════════════════════════════════════


class TestMarketOpenDuringHours:

    def test_10am_et_is_open(self, mh):
        # 10:00 AM ET = 15:00 UTC
        dt = _utc(2024, 3, 5, 15, 0)  # Tuesday
        assert mh.is_market_open(dt) is True

    def test_930am_et_is_open(self, mh):
        # 9:30 AM ET = 14:30 UTC
        dt = _utc(2024, 3, 5, 14, 30)  # Tuesday
        assert mh.is_market_open(dt) is True

    def test_359pm_et_is_open(self, mh):
        # 3:59 PM ET = 20:59 UTC
        dt = _utc(2024, 3, 5, 20, 59)  # Tuesday
        assert mh.is_market_open(dt) is True

    def test_trading_allowed_during_regular(self, mh):
        dt = _utc(2024, 3, 5, 15, 0)  # 10 AM ET Tue
        assert mh.is_trading_allowed(dt) is True


# ═══════════════════════════════════════════════════════════════════════
#  2. Market Closed in Evening
# ═══════════════════════════════════════════════════════════════════════


class TestMarketClosedEvening:

    def test_6pm_et_closed(self, mh):
        # 6:00 PM ET = 23:00 UTC
        dt = _utc(2024, 3, 5, 23, 0)  # Tuesday evening
        assert mh.is_market_open(dt) is False

    def test_4pm_et_is_closed(self, mh):
        # 4:00 PM ET = 21:00 UTC (at close boundary → closed)
        dt = _utc(2024, 3, 5, 21, 0)  # Tuesday
        assert mh.is_market_open(dt) is False

    def test_trading_not_allowed_evening(self, mh):
        dt = _utc(2024, 3, 5, 23, 0)  # 6 PM ET
        assert mh.is_trading_allowed(dt) is False


# ═══════════════════════════════════════════════════════════════════════
#  3. Market Closed on Weekend
# ═══════════════════════════════════════════════════════════════════════


class TestMarketClosedWeekend:

    def test_saturday_closed(self, mh):
        dt = _utc(2024, 3, 9, 15, 0)  # Saturday
        assert mh.is_market_open(dt) is False

    def test_sunday_closed(self, mh):
        dt = _utc(2024, 3, 10, 15, 0)  # Sunday
        assert mh.is_market_open(dt) is False

    def test_trading_not_allowed_weekend(self, mh):
        dt = _utc(2024, 3, 9, 15, 0)  # Saturday
        assert mh.is_trading_allowed(dt) is False


# ═══════════════════════════════════════════════════════════════════════
#  4. Premarket Allowed
# ═══════════════════════════════════════════════════════════════════════


class TestPremarketAllowed:

    def test_5am_et_allowed_with_premarket(self, mh_extended):
        # 5:00 AM ET = 10:00 UTC
        dt = _utc(2024, 3, 5, 10, 0)  # Tuesday
        assert mh_extended.is_trading_allowed(dt) is True

    def test_premarket_boundary_4am_et(self, mh_extended):
        # 4:00 AM ET = 09:00 UTC
        dt = _utc(2024, 3, 5, 9, 0)
        assert mh_extended.is_trading_allowed(dt) is True


# ═══════════════════════════════════════════════════════════════════════
#  5. Premarket Blocked
# ═══════════════════════════════════════════════════════════════════════


class TestPremarketBlocked:

    def test_5am_et_blocked_without_premarket(self, mh):
        # 5:00 AM ET = 10:00 UTC
        dt = _utc(2024, 3, 5, 10, 0)
        assert mh.is_trading_allowed(dt) is False

    def test_8am_et_blocked_without_premarket(self, mh):
        # 8:00 AM ET = 13:00 UTC
        dt = _utc(2024, 3, 5, 13, 0)
        assert mh.is_trading_allowed(dt) is False


# ═══════════════════════════════════════════════════════════════════════
#  6. Should Flatten EOD
# ═══════════════════════════════════════════════════════════════════════


class TestShouldFlattenEOD:

    def test_345pm_et_should_flatten(self, mh):
        # 3:45 PM ET = 20:45 UTC → 15 min to close → True
        dt = _utc(2024, 3, 5, 20, 45)
        assert mh.should_flatten_eod(dt) is True

    def test_350pm_et_should_flatten(self, mh):
        # 3:50 PM ET = 20:50 UTC → 10 min to close → True
        dt = _utc(2024, 3, 5, 20, 50)
        assert mh.should_flatten_eod(dt) is True

    def test_2pm_et_should_not_flatten(self, mh):
        # 2:00 PM ET = 19:00 UTC → 2 hours to close → False
        dt = _utc(2024, 3, 5, 19, 0)
        assert mh.should_flatten_eod(dt) is False

    def test_flatten_when_market_closed_returns_false(self, mh):
        # 5 PM ET → market is closed → should not flatten
        dt = _utc(2024, 3, 5, 22, 0)
        assert mh.should_flatten_eod(dt) is False

    def test_custom_minutes_before_close(self, mh):
        # 3:30 PM ET with 30 min buffer → should flatten
        dt = _utc(2024, 3, 5, 20, 30)
        assert mh.should_flatten_eod(dt, minutes_before_close=30) is True


# ═══════════════════════════════════════════════════════════════════════
#  7. Time to Close
# ═══════════════════════════════════════════════════════════════════════


class TestTimeToClose:

    def test_time_to_close_during_hours(self, mh):
        # 2:00 PM ET = 19:00 UTC → 2 hours to close
        dt = _utc(2024, 3, 5, 19, 0)
        remaining = mh.time_to_close(dt)
        assert remaining.total_seconds() == pytest.approx(7200, abs=60)

    def test_time_to_close_after_close(self, mh):
        # 5:00 PM ET = 22:00 UTC → past close → 0
        dt = _utc(2024, 3, 5, 22, 0)
        remaining = mh.time_to_close(dt)
        assert remaining.total_seconds() == 0

    def test_time_to_close_at_open(self, mh):
        # 9:30 AM ET = 14:30 UTC → 6.5 hours to close
        dt = _utc(2024, 3, 5, 14, 30)
        remaining = mh.time_to_close(dt)
        assert remaining.total_seconds() == pytest.approx(6.5 * 3600, abs=60)


# ═══════════════════════════════════════════════════════════════════════
#  8. Holiday Detection
# ═══════════════════════════════════════════════════════════════════════


class TestHolidayDetection:

    def test_july_4th_2024_is_holiday(self, mh):
        assert mh.is_holiday(date(2024, 7, 4)) is True

    def test_christmas_2024_is_holiday(self, mh):
        assert mh.is_holiday(date(2024, 12, 25)) is True

    def test_regular_day_not_holiday(self, mh):
        assert mh.is_holiday(date(2024, 3, 5)) is False

    def test_market_closed_on_holiday(self, mh):
        # July 4th 2024 at 10 AM ET = 15:00 UTC (Thursday)
        dt = _utc(2024, 7, 4, 15, 0)
        assert mh.is_market_open(dt) is False

    def test_trading_blocked_on_holiday(self, mh):
        dt = _utc(2024, 7, 4, 15, 0)
        assert mh.is_trading_allowed(dt) is False

    def test_thanksgiving_2025_is_holiday(self, mh):
        assert mh.is_holiday(date(2025, 11, 27)) is True

    def test_mlk_day_2025_is_holiday(self, mh):
        assert mh.is_holiday(date(2025, 1, 20)) is True


# ═══════════════════════════════════════════════════════════════════════
#  9. Next Market Open
# ═══════════════════════════════════════════════════════════════════════


class TestNextMarketOpen:

    def test_next_open_from_friday_evening(self, mh):
        # Friday 5 PM ET = Sat 22:00 UTC (after market hours)
        dt = _utc(2024, 3, 8, 22, 0)
        next_open = mh.next_market_open(dt)
        # Should be Monday 9:30 AM ET
        assert next_open.weekday() == 0  # Monday

    def test_next_open_from_midday(self, mh):
        # Tuesday 2 PM ET → next day Wednesday 9:30 AM
        dt = _utc(2024, 3, 5, 19, 0)
        next_open = mh.next_market_open(dt)
        et = mh._to_eastern(next_open)
        assert et.hour == 9
        assert et.minute == 30


# ═══════════════════════════════════════════════════════════════════════
#  10. Afterhours
# ═══════════════════════════════════════════════════════════════════════


class TestAfterhours:

    def test_afterhours_allowed(self, mh_extended):
        # 5:00 PM ET = 22:00 UTC → afterhours with allow_afterhours=True
        dt = _utc(2024, 3, 5, 22, 0)
        assert mh_extended.is_trading_allowed(dt) is True

    def test_afterhours_blocked_default(self, mh):
        dt = _utc(2024, 3, 5, 22, 0)
        assert mh.is_trading_allowed(dt) is False
