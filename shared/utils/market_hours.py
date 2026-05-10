"""Market hours utilities for trading session management."""
from datetime import datetime, time, date, timezone, timedelta
from typing import Optional
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
    except ImportError:
        ZoneInfo = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class MarketHours:
    """Tracks market session hours and enforces trading windows."""

    # US market hours (Eastern Time)
    PREMARKET_OPEN = time(4, 0)       # 4:00 AM ET
    MARKET_OPEN = time(9, 30)          # 9:30 AM ET
    MARKET_CLOSE = time(16, 0)         # 4:00 PM ET
    AFTERHOURS_CLOSE = time(20, 0)     # 8:00 PM ET

    # Major US market holidays for 2024-2027
    US_HOLIDAYS_2024_2025 = [
        # 2024
        date(2024, 1, 1),    # New Year's Day
        date(2024, 1, 15),   # MLK Day
        date(2024, 2, 19),   # Presidents' Day
        date(2024, 3, 29),   # Good Friday
        date(2024, 5, 27),   # Memorial Day
        date(2024, 6, 19),   # Juneteenth
        date(2024, 7, 4),    # Independence Day
        date(2024, 9, 2),    # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
        # 2025
        date(2025, 1, 1),    # New Year's Day
        date(2025, 1, 20),   # MLK Day
        date(2025, 2, 17),   # Presidents' Day
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 26),   # Memorial Day
        date(2025, 6, 19),   # Juneteenth
        date(2025, 7, 4),    # Independence Day
        date(2025, 9, 1),    # Labor Day
        date(2025, 11, 27),  # Thanksgiving
        date(2025, 12, 25),  # Christmas
        # 2026
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # MLK Day (3rd Monday Jan)
        date(2026, 2, 16),   # Presidents' Day (3rd Monday Feb)
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day (last Monday May)
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day (observed — Jul 4 is Sat)
        date(2026, 9, 7),    # Labor Day (1st Monday Sep)
        date(2026, 11, 26),  # Thanksgiving (4th Thursday Nov)
        date(2026, 12, 25),  # Christmas
        # 2027
        date(2027, 1, 1),    # New Year's Day
        date(2027, 1, 18),   # MLK Day (3rd Monday Jan)
        date(2027, 2, 15),   # Presidents' Day (3rd Monday Feb)
        date(2027, 3, 26),   # Good Friday
        date(2027, 5, 31),   # Memorial Day (last Monday May)
        date(2027, 6, 18),   # Juneteenth (observed — Jun 19 is Sat)
        date(2027, 7, 5),    # Independence Day (observed — Jul 4 is Sun)
        date(2027, 9, 6),    # Labor Day (1st Monday Sep)
        date(2027, 11, 25),  # Thanksgiving (4th Thursday Nov)
        date(2027, 12, 24),  # Christmas (observed — Dec 25 is Sat)
    ]

    def __init__(
        self,
        allow_premarket: bool = False,
        allow_afterhours: bool = False,
        timezone_offset: int = -5,
    ):
        self.allow_premarket = allow_premarket
        self.allow_afterhours = allow_afterhours
        self.tz_offset = timezone_offset

    def _to_eastern(self, now: datetime = None) -> datetime:
        """Convert to US Eastern time with automatic DST handling.

        Uses zoneinfo.ZoneInfo("America/New_York") for correct EDT/EST
        transitions. Falls back to the configured fixed UTC offset only
        when zoneinfo is unavailable.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        if ZoneInfo is not None:
            return now.astimezone(ZoneInfo("America/New_York"))

        eastern = timezone(timedelta(hours=self.tz_offset))
        return now.astimezone(eastern)

    def is_market_open(self, now: datetime = None) -> bool:
        """Check if regular market is currently open."""
        et = self._to_eastern(now)
        if et.weekday() >= 5:
            return False
        if self.is_holiday(et.date()):
            return False
        current_time = et.time()
        return self.MARKET_OPEN <= current_time < self.MARKET_CLOSE

    def is_trading_allowed(self, now: datetime = None) -> bool:
        """Check if trading is allowed based on configured session."""
        et = self._to_eastern(now)
        if et.weekday() >= 5:
            return False
        if self.is_holiday(et.date()):
            return False
        current_time = et.time()

        if self.is_market_open(now):
            return True
        if self.allow_premarket and self.PREMARKET_OPEN <= current_time < self.MARKET_OPEN:
            return True
        if self.allow_afterhours and self.MARKET_CLOSE <= current_time < self.AFTERHOURS_CLOSE:
            return True
        return False

    def time_to_close(self, now: datetime = None) -> timedelta:
        """Returns time remaining until market close."""
        et = self._to_eastern(now)
        close_dt = et.replace(
            hour=self.MARKET_CLOSE.hour,
            minute=self.MARKET_CLOSE.minute,
            second=0,
            microsecond=0,
        )
        remaining = close_dt - et
        if remaining.total_seconds() < 0:
            return timedelta(0)
        return remaining

    def should_flatten_eod(self, now: datetime = None, minutes_before_close: int = 15) -> bool:
        """Returns True if positions should be flattened before close."""
        if not self.is_market_open(now):
            return False
        remaining = self.time_to_close(now)
        return remaining <= timedelta(minutes=minutes_before_close)

    def is_holiday(self, check_date: date = None) -> bool:
        """Check if a date is a US market holiday.

        Uses the static list for 2024-2027 and falls back to dynamic
        calculation for years outside that range.
        """
        if check_date is None:
            check_date = self._to_eastern().date()
        if check_date in self.US_HOLIDAYS_2024_2025:
            return True
        # Dynamic calculation for years beyond the static list
        return self._is_dynamic_holiday(check_date)

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Return the nth occurrence of a weekday in a month.

        Args:
            year: Calendar year.
            month: Month (1-12).
            weekday: 0=Monday ... 6=Sunday.
            n: Which occurrence (1-based). Use -1 for last.
        """
        if n > 0:
            first = date(year, month, 1)
            offset = (weekday - first.weekday()) % 7
            result = first + timedelta(days=offset + 7 * (n - 1))
            return result
        else:
            # Last occurrence
            if month == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month + 1, 1) - timedelta(days=1)
            offset = (last_day.weekday() - weekday) % 7
            return last_day - timedelta(days=offset)

    @staticmethod
    def _observed(d: date) -> date:
        """Return the observed date for a fixed holiday.

        If the holiday falls on Saturday, observed on Friday.
        If it falls on Sunday, observed on Monday.
        """
        if d.weekday() == 5:
            return d - timedelta(days=1)
        elif d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    @staticmethod
    def _easter_date(year: int) -> date:
        """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
        a = year % 19
        b, c = divmod(year, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    @classmethod
    def _is_dynamic_holiday(cls, check_date: date) -> bool:
        """Dynamically compute US market holidays for any year."""
        year = check_date.year
        holidays = [
            cls._observed(date(year, 1, 1)),            # New Year's Day
            cls._nth_weekday(year, 1, 0, 3),             # MLK Day (3rd Mon Jan)
            cls._nth_weekday(year, 2, 0, 3),             # Presidents' Day (3rd Mon Feb)
            cls._nth_weekday(year, 5, 0, -1),            # Memorial Day (last Mon May)
            cls._observed(date(year, 6, 19)),             # Juneteenth
            cls._observed(date(year, 7, 4)),              # Independence Day
            cls._nth_weekday(year, 9, 0, 1),              # Labor Day (1st Mon Sep)
            cls._nth_weekday(year, 11, 3, 4),             # Thanksgiving (4th Thu Nov)
            cls._observed(date(year, 12, 25)),            # Christmas
        ]
        if check_date in holidays:
            return True

        # Good Friday (2 days before Easter Sunday)
        easter = cls._easter_date(check_date.year)
        good_friday = easter - timedelta(days=2)
        if check_date == good_friday:
            return True

        return False

    def next_market_open(self, now: datetime = None) -> datetime:
        """Returns the next market open datetime."""
        et = self._to_eastern(now)
        candidate = et.replace(
            hour=self.MARKET_OPEN.hour,
            minute=self.MARKET_OPEN.minute,
            second=0,
            microsecond=0,
        )
        if et.time() >= self.MARKET_OPEN:
            candidate += timedelta(days=1)

        while candidate.weekday() >= 5 or self.is_holiday(candidate.date()):
            candidate += timedelta(days=1)

        return candidate
