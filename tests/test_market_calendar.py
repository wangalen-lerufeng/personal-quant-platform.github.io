from datetime import datetime, time, timezone
import unittest

from quant_platform.market_calendar import is_trading_time


class MarketCalendarTests(unittest.TestCase):
    def test_mainland_equity_is_open_during_day_session(self):
        moment = datetime(2026, 1, 5, 2, 0, tzinfo=timezone.utc)

        self.assertTrue(is_trading_time("mainland_equity", moment))

    def test_mainland_equity_is_closed_at_lunch_break(self):
        moment = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)

        self.assertFalse(is_trading_time("mainland_equity", moment))

    def test_mt5_is_closed_on_weekend(self):
        moment = datetime(2026, 1, 10, 10, 0, tzinfo=timezone.utc)

        self.assertFalse(is_trading_time("mt5_broker_symbols", moment))

    def test_mainland_futures_is_open_in_day_session(self):
        moment = datetime(2026, 1, 5, 1, 30, tzinfo=timezone.utc)  # 09:30 Shanghai

        self.assertTrue(is_trading_time("mainland_futures", moment))

    def test_mainland_futures_is_open_in_night_session(self):
        moment = datetime(2026, 1, 5, 13, 30, tzinfo=timezone.utc)  # 21:30 Shanghai

        self.assertTrue(is_trading_time("mainland_futures", moment))

    def test_mainland_futures_is_closed_between_sessions(self):
        moment = datetime(2026, 1, 5, 3, 30, tzinfo=timezone.utc)  # 11:30 Shanghai

        self.assertFalse(is_trading_time("mainland_futures", moment))
