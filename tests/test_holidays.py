from datetime import date, datetime, timezone
import unittest

from quant_platform.market_calendar import MarketCalendar


class HolidayTests(unittest.TestCase):
    def test_configured_holiday_is_closed(self):
        calendar = MarketCalendar(holidays={date(2026, 1, 1)})

        self.assertFalse(calendar.is_open("mainland_equity", datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)))

    def test_unconfigured_weekday_uses_normal_session(self):
        calendar = MarketCalendar()

        self.assertTrue(calendar.is_open("mainland_equity", datetime(2026, 1, 5, 2, 0, tzinfo=timezone.utc)))
