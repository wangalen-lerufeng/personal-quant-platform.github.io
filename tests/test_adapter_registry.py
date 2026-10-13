import unittest
from datetime import datetime, timezone

from quant_platform.adapters import CtpAdapter, Mt5Adapter, QmtAdapter
from quant_platform.adapter_registry import AdapterRegistry
from quant_platform.domain import AssetClass, MarketEvent, OrderIntent, OrderSide
from quant_platform.market_data import InMemoryMarketDataFeed


class AdapterRegistryTests(unittest.TestCase):
    def test_registry_reports_all_market_adapters(self):
        registry = AdapterRegistry([QmtAdapter(), CtpAdapter(), Mt5Adapter()])

        statuses = registry.health()

        self.assertEqual([status["market"] for status in statuses], [
            "mainland_equity",
            "mainland_futures",
            "mt5_broker_symbols",
        ])
        self.assertTrue(all(status["status"] == "ready" for status in statuses))

    def test_registry_resolves_adapter_by_market(self):
        registry = AdapterRegistry([QmtAdapter(), CtpAdapter(), Mt5Adapter()])

        self.assertEqual(registry.for_market("mainland_futures").name, "ctp-paper")

    def test_registry_rejects_duplicate_market_registration(self):
        with self.assertRaises(ValueError):
            AdapterRegistry([QmtAdapter(), QmtAdapter()])

    def test_registry_routes_market_data_to_matching_adapter(self):
        registry = AdapterRegistry([QmtAdapter(), CtpAdapter(), Mt5Adapter()])
        cases = (
            ("mainland_equity", AssetClass.ETF, "510300", 3.5),
            ("mainland_futures", AssetClass.FUTURES, "IF2601", 3_500.0),
            ("mt5_broker_symbols", AssetClass.FX, "EURUSD", 1.08),
        )
        for market, asset_class, symbol, price in cases:
            with self.subTest(market=market):
                feed = InMemoryMarketDataFeed()
                received = []
                feed.subscribe(received.append)
                registry.publish_market_data(
                    market,
                    feed,
                    [MarketEvent(symbol, asset_class, price, datetime(2026, 1, 1, tzinfo=timezone.utc))],
                )
                self.assertEqual(received[0].symbol, symbol)

    def test_registry_routes_orders_with_market_asset_isolation(self):
        registry = AdapterRegistry([QmtAdapter(), CtpAdapter(), Mt5Adapter()])
        cases = (
            ("mainland_equity", AssetClass.ETF, "510300"),
            ("mainland_futures", AssetClass.FUTURES, "IF2601"),
            ("mt5_broker_symbols", AssetClass.FX, "EURUSD"),
        )
        for market, asset_class, symbol in cases:
            with self.subTest(market=market):
                receipt = registry.submit_order(
                    market,
                    OrderIntent(symbol, OrderSide.BUY, 1, "registry", asset_class=asset_class),
                )
                self.assertEqual(receipt.status, "accepted")
                self.assertEqual(registry.cancel_order(market, receipt.order_id).value, "canceled")
                self.assertEqual(registry.get_positions(market), {})

        rejected = registry.submit_order(
            "mainland_equity",
            OrderIntent("EURUSD", OrderSide.BUY, 1, "registry", asset_class=AssetClass.FX),
        )
        self.assertEqual(rejected.status, "rejected")
