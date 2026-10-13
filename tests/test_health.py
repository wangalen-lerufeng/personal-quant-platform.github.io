import unittest

from quant_platform.health import health_snapshot


class HealthTests(unittest.TestCase):
    def test_health_snapshot_is_safe_for_monitoring(self):
        snapshot = health_snapshot(environment="development", live_trading_enabled=False)

        self.assertEqual(snapshot["status"], "ok")
        self.assertFalse(snapshot["live_trading_enabled"])
        self.assertFalse(snapshot["ai_execution_tools_enabled"])
        self.assertFalse(snapshot["execution_enabled"])
        self.assertTrue(snapshot["paper_only"])

    def test_ai_execution_tools_disable_paper_only_status(self):
        snapshot = health_snapshot("development", False, True)

        self.assertFalse(snapshot["paper_only"])
        self.assertTrue(snapshot["execution_enabled"])
