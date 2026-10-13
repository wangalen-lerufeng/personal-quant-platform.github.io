import unittest

import pandas as pd

from quant_platform.factor_analysis import analyze_factor, analyze_factor_panel, winsorize_zscore


class FactorAnalysisTests(unittest.TestCase):
    def test_winsorize_and_standardize(self):
        values = pd.Series([1.0, 2.0, 3.0, 100.0])
        normalized = winsorize_zscore(values, limits=(0.25, 0.25))
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.std(ddof=0)), 1.0, places=6)

    def test_factor_report_contains_ic_and_group_returns(self):
        frame = pd.DataFrame({
            "factor": [1, 2, 3, 4, 5, 6],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        })
        report = analyze_factor(frame, "factor", "forward_return", groups=3)
        self.assertGreater(report["ic"], 0.99)
        self.assertGreater(report["rank_ic"], 0.99)
        self.assertEqual(len(report["group_returns"]), 3)

    def test_panel_report_contains_daily_ic_icir_decay_and_turnover(self):
        rows = []
        for day, returns in (
            ("2026-01-02", [0.01, 0.02, 0.03, 0.04]),
            ("2026-01-05", [0.04, 0.03, 0.02, 0.01]),
            ("2026-01-06", [0.01, 0.02, 0.03, 0.04]),
        ):
            for index, symbol in enumerate(("A", "B", "C", "D"), start=1):
                rows.append({
                    "timestamp": day,
                    "symbol": symbol,
                    "factor": index,
                    "forward_return_1d": returns[index - 1],
                    "forward_return_5d": returns[index - 1] / 2,
                    "dollar_volume": 1_000_000 * index,
                })

        report = analyze_factor_panel(
            pd.DataFrame(rows),
            factor_column="factor",
            return_columns=["forward_return_1d", "forward_return_5d"],
            groups=2,
            dollar_volume_column="dollar_volume",
            participation_rate=0.01,
        )

        self.assertEqual(report["date_count"], 3)
        self.assertEqual(len(report["ic_series"]["forward_return_1d"]), 3)
        self.assertAlmostEqual(report["horizons"]["forward_return_1d"]["ic_mean"], 1 / 3)
        self.assertAlmostEqual(report["horizons"]["forward_return_1d"]["icir"], 0.28867513459481287)
        self.assertEqual(set(report["decay"]), {"forward_return_1d", "forward_return_5d"})
        self.assertAlmostEqual(report["factor_autocorrelation"], 1.0)
        self.assertAlmostEqual(report["top_group_turnover"], 0.0)
        self.assertEqual(report["capacity"]["estimated_notional"], 70_000.0)
        self.assertEqual(report["alignment"], "factor_at_t_vs_forward_return_after_t")

    def test_panel_report_splits_in_sample_and_out_of_sample_chronologically(self):
        rows = []
        for day in ("2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"):
            for index, symbol in enumerate(("A", "B", "C", "D"), start=1):
                rows.append({
                    "timestamp": day,
                    "symbol": symbol,
                    "factor": index,
                    "forward_return": index / 100,
                })

        report = analyze_factor_panel(
            pd.DataFrame(rows),
            factor_column="factor",
            return_columns=["forward_return"],
            groups=2,
            out_of_sample_start="2026-01-06",
        )

        self.assertEqual(report["sample_split"]["in_sample"]["date_count"], 2)
        self.assertEqual(report["sample_split"]["out_of_sample"]["date_count"], 2)
        self.assertEqual(report["sample_split"]["out_of_sample_start"], "2026-01-06T00:00:00")

    def test_panel_report_rejects_non_forward_return_labels(self):
        frame = pd.DataFrame({
            "timestamp": ["2026-01-01"] * 4,
            "symbol": ["A", "B", "C", "D"],
            "factor": [1, 2, 3, 4],
            "return": [0.01, 0.02, 0.03, 0.04],
        })

        with self.assertRaisesRegex(ValueError, "forward-return"):
            analyze_factor_panel(frame, "factor", ["return"], groups=2)
