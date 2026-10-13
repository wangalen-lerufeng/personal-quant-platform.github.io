import unittest

from quant_platform.app import build_paper_app
from quant_platform.config import RuntimeConfig
from quant_platform.risk import RiskLimits


class AppFactoryTests(unittest.TestCase):
    def test_paper_app_wires_all_market_adapters_and_safe_flags(self):
        app = build_paper_app()

        self.assertFalse(app.config.live_trading_enabled)
        self.assertEqual(len(app.control_plane.adapter_health()), 3)
        self.assertNotIn("submit_order", app.research_tools.catalog())
        self.assertIsNotNone(app.simulation_registry)
        self.assertEqual(app.simulation_registry.list_states(), [])

    def test_paper_app_accepts_safe_runtime_and_risk_configuration(self):
        app = build_paper_app(
            config=RuntimeConfig(environment="test"),
            risk_limits=RiskLimits(max_order_value=10_000),
        )

        self.assertEqual(app.config.environment, "test")
        self.assertEqual(app.control_plane.risk_engine.limits.max_order_value, 10_000)

    def test_paper_app_rejects_live_runtime_configuration(self):
        with self.assertRaises(ValueError):
            build_paper_app(config=RuntimeConfig(live_trading_enabled=True))
