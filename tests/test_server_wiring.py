import unittest
from datetime import datetime, timezone

from quant_platform.app import build_paper_app
from quant_platform.http_api import api_payload, make_handler
from quant_platform.domain import AssetClass, MarketEvent, OrderSide


class ServerWiringTests(unittest.TestCase):
    def test_handler_factory_receives_control_plane(self):
        app = build_paper_app()
        handler = make_handler(app.config, app.control_plane, app.research_tools)

        self.assertIsNotNone(handler)
        self.assertEqual(api_payload("/adapters", app.config, app.control_plane)["adapters"][0]["mode"], "paper")
        self.assertFalse(api_payload("/research", app.config, app.control_plane, app.research_tools)["execution_enabled"])

    def test_paper_app_portfolio_route_marks_to_latest_feed_price(self):
        app = build_paper_app()
        app.control_plane.record_fill("EURUSD", OrderSide.BUY, 0.01, 1.08)
        app.market_data_feed.publish(
            MarketEvent("EURUSD", AssetClass.FX, 1.09, datetime(2026, 1, 1, tzinfo=timezone.utc))
        )

        payload = api_payload("/portfolio", app.config, app.control_plane, app.research_tools)

        self.assertAlmostEqual(payload["positions"]["EURUSD"]["market_price"], 1.09)
        self.assertAlmostEqual(payload["unrealized_pnl"], 0.0001)
