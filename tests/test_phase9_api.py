import unittest
from unittest.mock import patch

from quant_platform.http_api import phase9_payload
from quant_platform.simulation import PaperSimulationRegistry
from quant_platform.storage import SimulationAccountStore

SYMBOLS = ["510300.SH", "510500.SH", "159919.SZ", "512880.SH", "515790.SH"]


def _fake_research():
    return {
        "panel": {"rows": 10, "symbols": 5, "start": "2026-01-01", "end": "2026-06-01"},
        "baseline": {"rank_ic": 0.5, "icir": 1.0},
        "ensemble": {"rank_ic": 0.5, "icir": 1.0},
        "candidates": ["510300.SH", "510500.SH"],
        "verification": [],
        "sizing": {"adjustment": 1.0},
        "targets": [
            {"symbol": "510300.SH", "weight": 0.5},
            {"symbol": "510500.SH", "weight": 0.5},
        ],
        "research_only": True,
        "execution_enabled": False,
    }


class Phase9ApiTests(unittest.TestCase):
    def test_phase9_payload_submits_to_paper_session(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        with patch("quant_platform.http_api.run_phase9_alpha158", return_value=_fake_research()):
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
        self.assertIn("account_id", result)
        self.assertEqual(result["account_id"], result["run_id"])
        session = registry.get(result["account_id"])
        self.assertTrue(session.state()["orders"])

    def test_phase9_payload_creates_distinct_accounts_per_run(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        with patch("quant_platform.http_api.run_phase9_alpha158", return_value=_fake_research()):
            first = phase9_payload(
                {"symbols": SYMBOLS, "top_k": 3, "start_date": "2026-01-01", "end_date": "2026-06-01"},
                None,
                registry,
            )
            second = phase9_payload(
                {"symbols": SYMBOLS, "top_k": 3, "start_date": "2026-01-01", "end_date": "2026-06-01"},
                None,
                registry,
            )
        self.assertNotEqual(first["account_id"], second["account_id"])
        account_ids = {s["account_id"] for s in registry.list_states()}
        self.assertEqual(len(account_ids), 2)
        self.assertNotIn("paper-main", account_ids)

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
