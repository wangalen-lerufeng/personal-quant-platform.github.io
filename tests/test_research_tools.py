import unittest
from datetime import datetime, timezone

from quant_platform.config import RuntimeConfig
from quant_platform.control import ControlPlane
from quant_platform.domain import AssetClass, MarketEvent, OrderIntent, OrderSide
from quant_platform.market_data import InMemoryMarketDataFeed
from quant_platform.research_tools import ResearchTools
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.storage import SqliteAuditStore
from quant_platform.experiments import ExperimentStore
from quant_platform.portfolio import PositionBook


class ResearchToolsTests(unittest.TestCase):
    def setUp(self):
        store = SqliteAuditStore(":memory:")
        control = ControlPlane(RuntimeConfig(), RiskEngine(RiskLimits()), store)
        control.record_order("order-1", OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"), "accepted")
        experiments = ExperimentStore()
        experiments.save("demo", {"trade_count": 1, "total_return": 0.001})
        self.tools = ResearchTools(control, experiments)

    def test_tools_can_read_system_and_risk_state(self):
        self.assertFalse(self.tools.system_status()["live_trading_enabled"])
        self.assertFalse(self.tools.risk_status()["kill_switch"])

    def test_tools_can_list_orders_as_read_only_data(self):
        orders = self.tools.recent_orders()

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "EURUSD")

    def test_tools_can_read_append_only_audit_events(self):
        events = self.tools.audit_events("order-1")

        self.assertEqual(events[0]["event_type"], "order_recorded")
        self.assertEqual(events[0]["status"], "accepted")

    def test_tool_catalog_contains_no_execution_operation(self):
        self.assertNotIn("submit_order", self.tools.catalog())
        self.assertIn("recent_orders", self.tools.catalog())
        self.assertIn("run_backtest", self.tools.catalog())
        self.assertIn("analyze_factor", self.tools.catalog())
        self.assertIn("analyze_factor_panel", self.tools.catalog())
        self.assertIn("classify_market_state", self.tools.catalog())
        self.assertIn("monitor_simulation_risk", self.tools.catalog())
        self.assertNotIn("execute_order", self.tools.catalog())

    def test_tool_catalog_exposes_strategy_metadata_as_read_only(self):
        self.assertIn("strategies", self.tools.catalog())
        self.assertEqual(self.tools.strategies(), [])

    def test_tools_can_list_experiment_results(self):
        results = self.tools.experiment_results()

        self.assertEqual(results[0]["strategy_id"], "demo")

    def test_tools_can_read_portfolio_state_without_execution(self):
        state = self.tools.portfolio_state()

        self.assertEqual(state["positions"], {})
        self.assertEqual(state["equity"], 0.0)

    def test_market_data_tool_returns_standardized_events(self):
        event = MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc))
        tools = ResearchTools(self.tools.control_plane, ExperimentStore(), InMemoryMarketDataFeed([event]))

        rows = tools.market_data("EURUSD")

        self.assertEqual(rows[0]["symbol"], "EURUSD")
        self.assertEqual(rows[0]["price"], 1.08)

    def test_market_data_tool_can_limit_to_latest_rows(self):
        events = [
            MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc)),
            MarketEvent("EURUSD", AssetClass.FX, 1.09, datetime(2026, 1, 2, tzinfo=timezone.utc)),
        ]
        tools = ResearchTools(self.tools.control_plane, market_data_feed=InMemoryMarketDataFeed(events))

        rows = tools.market_data("EURUSD", limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["price"], 1.09)

    def test_market_data_tool_rejects_non_positive_limit(self):
        with self.assertRaises(ValueError):
            ResearchTools(self.tools.control_plane, market_data_feed=InMemoryMarketDataFeed()).market_data(limit=0)

    def test_portfolio_state_uses_latest_market_data_prices(self):
        position_book = PositionBook()
        control = ControlPlane(
            RuntimeConfig(), RiskEngine(RiskLimits()), SqliteAuditStore(":memory:"), position_book=position_book
        )
        control.record_fill("EURUSD", OrderSide.BUY, 10, 1.08)
        feed = InMemoryMarketDataFeed([
            MarketEvent("EURUSD", AssetClass.FX, 1.09, datetime(2026, 1, 1, tzinfo=timezone.utc)),
        ])
        tools = ResearchTools(control, market_data_feed=feed)

        snapshot = tools.portfolio_state()

        self.assertAlmostEqual(snapshot["positions"]["EURUSD"]["market_price"], 1.09)
        self.assertAlmostEqual(snapshot["unrealized_pnl"], 0.1)
