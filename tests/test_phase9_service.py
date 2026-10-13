import unittest

from quant_platform.phase9_service import run_phase9, synthetic_etf_panel

SYMBOLS = ["510300.SH", "510500.SH", "159919.SZ", "512880.SH", "515790.SH"]


class SyntheticPanelTests(unittest.TestCase):
    def test_panel_requires_two_symbols(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            synthetic_etf_panel(["510300.SH"])

    def test_panel_has_expected_columns(self):
        panel = synthetic_etf_panel(SYMBOLS[:3])
        self.assertIn("adj_close", panel.columns)
        self.assertIn("forward_return_5d", panel.columns)
        self.assertEqual(panel["symbol"].nunique(), 3)


class RunPhase9Tests(unittest.TestCase):
    def test_returns_read_only_result(self):
        result = run_phase9(SYMBOLS, top_k=3, seed=42)
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["research_only"])
        self.assertIn("panel", result)
        self.assertIn("baseline", result)
        self.assertIn("ensemble", result)

    def test_produces_candidates_verification_and_targets(self):
        result = run_phase9(SYMBOLS, top_k=3, seed=42)
        self.assertTrue(result["candidates"])
        self.assertTrue(result["verification"])
        self.assertTrue(result["targets"])
        passed_symbols = {item["symbol"] for item in result["verification"] if item["verdict"] == "pass"}
        target_symbols = {item["symbol"] for item in result["targets"]}
        self.assertLessEqual(target_symbols, passed_symbols)

    def test_is_deterministic(self):
        first = run_phase9(SYMBOLS, top_k=3, seed=42)
        second = run_phase9(SYMBOLS, top_k=3, seed=42)
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual(first["targets"], second["targets"])
        self.assertEqual(first["sizing"], second["sizing"])

    def test_rejects_insufficient_symbols(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            run_phase9(["510300.SH"])


if __name__ == "__main__":
    unittest.main()
