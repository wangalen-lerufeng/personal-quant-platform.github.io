import unittest

from quant_platform.config import RuntimeConfig
from quant_platform.control import ControlPlane
from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.storage import SqliteAuditStore
from quant_platform.adapter_registry import AdapterRegistry
from quant_platform.adapters import CtpAdapter, Mt5Adapter, QmtAdapter
from quant_platform.portfolio import PositionBook
from quant_platform.market_data import InMemoryMarketDataFeed
from quant_platform.domain import AssetClass, MarketEvent
from datetime import datetime, timezone


class ControlPlaneTests(unittest.TestCase):
    def test_control_plane_exposes_orders_and_risk_state(self):
        store = SqliteAuditStore(":memory:")
        plane = ControlPlane(
            RuntimeConfig(),
            RiskEngine(RiskLimits()),
            store,
            AdapterRegistry([QmtAdapter(), CtpAdapter(), Mt5Adapter()]),
        )

        plane.record_order(
            order_id="order-1",
            intent=OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"),
            status="pending",
        )

        self.assertEqual(plane.orders()[0]["order_id"], "order-1")
        self.assertFalse(plane.risk_state()["kill_switch"])
        self.assertFalse(plane.system_state()["live_trading_enabled"])
        self.assertFalse(plane.system_state()["execution_enabled"])
        self.assertTrue(plane.system_state()["paper_only"])
        self.assertEqual(len(plane.adapter_health()), 3)

    def test_control_plane_records_fill_into_position_book(self):
        book = PositionBook()
        plane = ControlPlane(RuntimeConfig(), RiskEngine(RiskLimits()), SqliteAuditStore(":memory:"), position_book=book)

        plane.record_fill("EURUSD", OrderSide.BUY, 0.01, 1.08)

        self.assertEqual(book.quantity("EURUSD"), 0.01)
        self.assertAlmostEqual(plane.portfolio_state({"EURUSD": 1.09})["unrealized_pnl"], 0.0001)
        self.assertEqual(plane.risk_state()["positions"]["EURUSD"], 0.01)

    def test_control_plane_uses_feed_latest_price_for_portfolio(self):
        book = PositionBook()
        feed = InMemoryMarketDataFeed([
            MarketEvent("EURUSD", AssetClass.FX, 1.09, datetime(2026, 1, 1, tzinfo=timezone.utc))
        ])
        plane = ControlPlane(
            RuntimeConfig(), RiskEngine(RiskLimits()), SqliteAuditStore(":memory:"),
            position_book=book, market_data_feed=feed,
        )
        plane.record_fill("EURUSD", OrderSide.BUY, 0.01, 1.08)

        snapshot = plane.portfolio_state()

        self.assertAlmostEqual(snapshot["positions"]["EURUSD"]["market_price"], 1.09)

    def test_control_plane_exposes_loss_and_drawdown_thresholds(self):
        plane = ControlPlane(
            RuntimeConfig(),
            RiskEngine(RiskLimits(max_daily_loss=500, max_drawdown=1000)),
            SqliteAuditStore(":memory:"),
        )

        state = plane.risk_state()

        self.assertEqual(state["max_daily_loss"], 500)
        self.assertEqual(state["max_drawdown"], 1000)
