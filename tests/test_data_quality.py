from datetime import datetime, timedelta, timezone
import unittest

from quant_platform.data_quality import DataQualityError, validate_events
from quant_platform.domain import AssetClass, MarketEvent


class DataQualityTests(unittest.TestCase):
    def _event(self, offset: int, price: float = 1.08):
        return MarketEvent("EURUSD", AssetClass.FX, price, datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset))

    def test_valid_events_pass_validation(self):
        self.assertEqual(validate_events([self._event(0), self._event(1)]), 2)

    def test_duplicate_timestamp_is_rejected(self):
        with self.assertRaisesRegex(DataQualityError, "duplicate"):
            validate_events([self._event(0), self._event(0)])

    def test_out_of_order_timestamp_is_rejected(self):
        with self.assertRaisesRegex(DataQualityError, "order"):
            validate_events([self._event(1), self._event(0)])

    def test_interleaved_symbols_are_valid_when_each_symbol_is_ordered(self):
        first = self._event(0)
        second = MarketEvent("USDJPY", AssetClass.FX, 150.0, first.timestamp)
        third = self._event(1)

        self.assertEqual(validate_events([first, second, third]), 3)

    def test_symbol_cannot_change_asset_class(self):
        first = self._event(0)
        second = MarketEvent("EURUSD", AssetClass.ETF, 1.08, first.timestamp + timedelta(days=1))

        with self.assertRaisesRegex(DataQualityError, "asset class"):
            validate_events([first, second])
