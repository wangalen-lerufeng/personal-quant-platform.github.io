from datetime import datetime, timezone
import unittest

from quant_platform.domain import AssetClass, MarketEvent
from quant_platform.data_quality import DataQualityError
from quant_platform.market_data import InMemoryMarketDataFeed


class MarketDataTests(unittest.TestCase):
    def test_feed_publishes_events_in_order(self):
        feed = InMemoryMarketDataFeed()
        first = MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc))
        second = MarketEvent("EURUSD", AssetClass.FX, 1.081, datetime(2026, 1, 2, tzinfo=timezone.utc))
        received = []

        feed.subscribe(received.append)
        feed.publish(first)
        feed.publish(second)

        self.assertEqual(received, [first, second])

    def test_feed_replays_events_to_new_subscriber(self):
        feed = InMemoryMarketDataFeed([MarketEvent("510300", AssetClass.ETF, 3.5, datetime(2026, 1, 1, tzinfo=timezone.utc))])
        received = []

        feed.subscribe(received.append, replay=True)

        self.assertEqual(len(received), 1)

    def test_feed_rejects_duplicate_event(self):
        event = MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc))
        feed = InMemoryMarketDataFeed()
        feed.publish(event)

        with self.assertRaises(DataQualityError):
            feed.publish(event)

    def test_feed_rejects_invalid_initial_history(self):
        first = MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 2, tzinfo=timezone.utc))
        earlier = MarketEvent("EURUSD", AssetClass.FX, 1.07, datetime(2026, 1, 1, tzinfo=timezone.utc))

        with self.assertRaises(DataQualityError):
            InMemoryMarketDataFeed([first, earlier])

    def test_feed_can_unsubscribe_callback(self):
        feed = InMemoryMarketDataFeed()
        received = []
        callback = received.append
        feed.subscribe(callback)
        feed.unsubscribe(callback)

        feed.publish(MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc)))

        self.assertEqual(received, [])

    def test_feed_subscription_is_idempotent(self):
        feed = InMemoryMarketDataFeed()
        received = []
        callback = received.append
        feed.subscribe(callback)
        feed.subscribe(callback)

        feed.publish(MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc)))

        self.assertEqual(len(received), 1)

    def test_callback_can_unsubscribe_without_skipping_other_callbacks(self):
        feed = InMemoryMarketDataFeed()
        received = []

        def first(event):
            received.append("first")
            feed.unsubscribe(first)

        def second(event):
            received.append("second")

        feed.subscribe(first)
        feed.subscribe(second)
        feed.publish(MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc)))

        self.assertEqual(received, ["first", "second"])

    def test_latest_prices_returns_last_event_per_symbol(self):
        feed = InMemoryMarketDataFeed([
            MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc)),
            MarketEvent("EURUSD", AssetClass.FX, 1.081, datetime(2026, 1, 2, tzinfo=timezone.utc)),
        ])

        self.assertEqual(feed.latest_prices(), {"EURUSD": 1.081})
