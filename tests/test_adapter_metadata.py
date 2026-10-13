import unittest

from quant_platform.adapters.ctp import CtpAdapter
from quant_platform.adapters.mt5 import Mt5Adapter
from quant_platform.adapters.qmt import QmtAdapter
from quant_platform.domain import AssetClass, MarketEvent, OrderIntent, OrderSide
from quant_platform.market_data import InMemoryMarketDataFeed
from quant_platform.execution import ExecutionGateway
from datetime import datetime, timezone


class AdapterMetadataTests(unittest.TestCase):
    def test_each_market_adapter_exposes_market_identity(self):
        self.assertEqual(QmtAdapter().market, "mainland_equity")
        self.assertEqual(CtpAdapter().market, "mainland_futures")
        self.assertEqual(Mt5Adapter().market, "mt5_broker_symbols")

    def test_each_adapter_starts_in_paper_mode(self):
        for adapter in (QmtAdapter(), CtpAdapter(), Mt5Adapter()):
            with self.subTest(adapter=adapter.__class__.__name__):
                self.assertEqual(adapter.mode, "paper")
                self.assertFalse(adapter.live_trading_enabled)
                self.assertIsInstance(adapter, ExecutionGateway)
                order = adapter.submit_order(OrderIntent("TEST", OrderSide.BUY, 1, "health"))
                self.assertEqual(adapter.cancel_order(order.order_id).value, "canceled")
                self.assertEqual(adapter.get_positions(), {})

    def test_health_check_reports_ready_without_live_connection(self):
        status = Mt5Adapter().health_check()

        self.assertEqual(status["status"], "ready")
        self.assertFalse(status["live_trading_enabled"])
        self.assertFalse(status["execution_enabled"])
        self.assertEqual(status["gateway"], "mt5-paper")
        self.assertEqual(status["supported_asset_class"], "fx")

    def test_adapter_rejects_order_for_another_asset_class(self):
        receipt = QmtAdapter().submit_order(
            OrderIntent("EURUSD", OrderSide.BUY, 1, "demo", asset_class=AssetClass.FX)
        )
        self.assertEqual(receipt.status, "rejected")

    def test_adapter_accepts_matching_asset_class(self):
        receipt = CtpAdapter().submit_order(
            OrderIntent("IF2601", OrderSide.BUY, 1, "demo", asset_class=AssetClass.FUTURES)
        )
        self.assertEqual(receipt.status, "accepted")

    def test_each_adapter_can_publish_matching_paper_market_data(self):
        cases = (
            (QmtAdapter(), AssetClass.ETF, "510300"),
            (CtpAdapter(), AssetClass.FUTURES, "IF2601"),
            (Mt5Adapter(), AssetClass.FX, "EURUSD"),
        )
        for adapter, asset_class, symbol in cases:
            with self.subTest(adapter=adapter.name):
                feed = InMemoryMarketDataFeed()
                received = []
                feed.subscribe(received.append)
                count = adapter.publish_market_data(
                    feed,
                    [MarketEvent(symbol, asset_class, 1.0, datetime(2026, 1, 1, tzinfo=timezone.utc))],
                )
                self.assertEqual(count, 1)
                self.assertEqual(received[0].symbol, symbol)

    def test_adapter_rejects_mixed_asset_batch_atomically(self):
        adapter = QmtAdapter()
        feed = InMemoryMarketDataFeed()
        received = []
        feed.subscribe(received.append)

        with self.assertRaises(ValueError):
            adapter.publish_market_data(
                feed,
                [
                    MarketEvent("510300", AssetClass.ETF, 3.5, datetime(2026, 1, 1, tzinfo=timezone.utc)),
                    MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc)),
                ],
            )

        self.assertEqual(received, [])
