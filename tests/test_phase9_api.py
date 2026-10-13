import unittest
from unittest.mock import patch

from quant_platform.http_api import phase9_payload
from quant_platform.phase9_service import synthetic_etf_panel
from quant_platform.simulation import PaperSimulationRegistry
from quant_platform.storage import SimulationAccountStore

SYMBOLS = ["510300.SH", "510500.SH", "159919.SZ", "512880.SH", "515790.SH"]


class Phase9ApiTests(unittest.TestCase):
    def test_phase9_payload_submits_to_paper_session(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        panel = synthetic_etf_panel(SYMBOLS)
        with patch("quant_platform.http_api.build_tushare_etf_panel", return_value=panel):
            result = phase9_payload(
                {"symbols": SYMBOLS, "top_k": 3, "start_date": "2026-01-01", "end_date": "2026-06-01"},
                None,
                registry,
            )

        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["paper_only"])
        self.assertTrue(result["research_only"])
        self.assertIn("orders", result)
        self.assertIn("rejected", result)
        self.assertIn("submission", result)

        self.assertGreater(result["submission"]["order_count"], 0)
        self.assertEqual(result["submission"]["submitted"][0]["status"], "accepted")
        session = registry.get("paper-main")
        self.assertTrue(session.state()["orders"])

    def test_phase9_payload_requires_symbols(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        with self.assertRaisesRegex(ValueError, "symbols"):
            phase9_payload({"symbols": [], "start_date": "2026-01-01"}, None, registry)

    def test_phase9_payload_requires_start_date(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        with self.assertRaisesRegex(ValueError, "start_date"):
            phase9_payload({"symbols": SYMBOLS, "start_date": ""}, None, registry)


if __name__ == "__main__":
    unittest.main()
