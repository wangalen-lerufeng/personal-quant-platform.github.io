import unittest

from quant_platform.cli import run_check


class CliTests(unittest.TestCase):
    def test_run_check_reports_safe_paper_configuration(self):
        result = run_check()

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["live_trading_enabled"])
        self.assertFalse(result["ai_execution_tools_enabled"])
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["paper_only"])
        self.assertEqual(len(result["adapters"]), 3)
        self.assertTrue(all(not adapter["execution_enabled"] for adapter in result["adapters"]))
