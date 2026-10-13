import unittest
from unittest.mock import patch

from quant_platform.phase9_service import (
    build_tushare_etf_panel,
    run_phase9,
    synthetic_etf_panel,
)

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


class BuildTusharePanelTests(unittest.TestCase):
    def _fake_records(self, dataset, symbol, start_date, end_date):
        dates = ["20260105", "20260106", "20260107", "20260108", "20260109"]
        if dataset == "fund_daily":
            return [
                {"trade_date": d, "close": 1.0 + i * 0.01, "vol": 1000, "amount": 100000}
                for i, d in enumerate(dates)
            ]
        if dataset == "fund_adj":
            return [{"trade_date": d, "adj_factor": 1.0} for d in dates]
        return []

    def test_build_tushare_etf_panel(self):
        with patch("quant_platform.phase9_service.TushareDatasetProvider") as MockProvider:
            MockProvider.return_value.fetch_records.side_effect = self._fake_records
            panel = build_tushare_etf_panel(
                ["510300.SH", "510500.SH"], "2026-01-05", "2026-01-09",
            )
        self.assertIn("adj_close", panel.columns)
        self.assertIn("forward_return_5d", panel.columns)
        self.assertEqual(panel["symbol"].nunique(), 2)

    def test_build_tushare_etf_panel_requires_dates(self):
        with self.assertRaisesRegex(ValueError, "required"):
            build_tushare_etf_panel(["510300.SH", "510500.SH"], "", "")


class RunPhase9Tests(unittest.TestCase):
    def test_returns_read_only_result(self):
        result = run_phase9(synthetic_etf_panel(SYMBOLS), top_k=3, seed=42)
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["research_only"])
        self.assertIn("panel", result)
        self.assertIn("baseline", result)
        self.assertIn("ensemble", result)

    def test_produces_candidates_verification_and_targets(self):
        result = run_phase9(synthetic_etf_panel(SYMBOLS), top_k=3, seed=42)
        self.assertTrue(result["candidates"])
        self.assertTrue(result["verification"])
        self.assertTrue(result["targets"])
        passed_symbols = {item["symbol"] for item in result["verification"] if item["verdict"] == "pass"}
        target_symbols = {item["symbol"] for item in result["targets"]}
        self.assertLessEqual(target_symbols, passed_symbols)

    def test_is_deterministic(self):
        first = run_phase9(synthetic_etf_panel(SYMBOLS), top_k=3, seed=42)
        second = run_phase9(synthetic_etf_panel(SYMBOLS), top_k=3, seed=42)
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual(first["targets"], second["targets"])
        self.assertEqual(first["sizing"], second["sizing"])

    def test_rejects_panel_with_insufficient_symbols(self):
        panel = synthetic_etf_panel(SYMBOLS)
        single = panel[panel["symbol"] == SYMBOLS[0]].copy()
        with self.assertRaisesRegex(ValueError, "at least two"):
            run_phase9(single)


if __name__ == "__main__":
    unittest.main()
