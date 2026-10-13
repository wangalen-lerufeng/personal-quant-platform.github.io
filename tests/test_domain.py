from datetime import datetime, timezone
import unittest

from quant_platform.domain import AssetClass, MarketEvent, OrderIntent, OrderSide, Signal


class DomainTests(unittest.TestCase):
    def test_market_event_requires_timezone_aware_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            MarketEvent(symbol="EURUSD", asset_class=AssetClass.FX, price=1.08, timestamp=datetime.now())

    def test_signal_can_be_converted_to_order_intent(self):
        signal = Signal(symbol="EURUSD", side=OrderSide.BUY, quantity=0.01, reason="trend")
        intent = OrderIntent.from_signal(signal, strategy_id="demo-trend")
        self.assertEqual(intent.symbol, "EURUSD")
        self.assertIs(intent.side, OrderSide.BUY)
        self.assertEqual(intent.strategy_id, "demo-trend")

    def test_market_event_accepts_aware_timestamp(self):
        event = MarketEvent(symbol="510300", asset_class=AssetClass.ETF, price=3.5, timestamp=datetime.now(timezone.utc))
        self.assertEqual(event.symbol, "510300")
