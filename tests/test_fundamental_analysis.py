from datetime import date
import unittest

from quant_platform.fundamental_analysis import analyze_fundamental_datasets


class FundamentalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {
                "dataset": "income",
                "data": [
                    {"end_date": "20240331", "ann_date": "20240420", "update_flag": "1", "revenue": 100.0, "n_income_attr_p": 10.0, "assets_impair_loss": -2.0},
                    {"end_date": "20240630", "ann_date": "20240801", "update_flag": "0", "revenue": 240.0, "n_income_attr_p": 20.0},
                    {"end_date": "20240630", "ann_date": "20240810", "update_flag": "1", "revenue": 250.0, "n_income_attr_p": 22.0},
                    {"end_date": "20240930", "ann_date": "20241025", "update_flag": "1", "revenue": 450.0, "n_income_attr_p": 35.0},
                    {"end_date": "20241231", "ann_date": "20250320", "update_flag": "1", "revenue": 700.0, "n_income_attr_p": 50.0},
                    {"end_date": "20251231", "ann_date": "20260318", "update_flag": "1", "revenue": 900.0, "n_income_attr_p": 70.0},
                ],
            },
            {
                "dataset": "cashflow",
                "data": [
                    {"end_date": "20240331", "ann_date": "20240420", "update_flag": "1", "n_cashflow_act": 8.0},
                    {"end_date": "20240630", "ann_date": "20240810", "update_flag": "1", "n_cashflow_act": 18.0},
                    {"end_date": "20240930", "ann_date": "20241025", "update_flag": "1", "n_cashflow_act": 30.0},
                    {"end_date": "20241231", "ann_date": "20250320", "update_flag": "1", "n_cashflow_act": 45.0},
                ],
            },
            {
                "dataset": "fina_indicator",
                "data": [
                    {"end_date": "20241231", "ann_date": "20250320", "update_flag": "1", "roe": 12.0, "roa": 8.0},
                ],
            },
        ]

    def test_deduplicates_disclosures_and_excludes_future_announcements(self):
        report = analyze_fundamental_datasets(self.results, date(2025, 12, 31))
        income = report["normalized"]["income"]
        self.assertEqual(len(income), 4)
        self.assertEqual(income[1]["revenue"], 250.0)
        self.assertNotIn("20251231", [row["end_date"] for row in income])

    def test_converts_cumulative_income_to_standalone_quarters(self):
        report = analyze_fundamental_datasets(self.results, date(2025, 12, 31))
        quarters = report["single_quarter"]["income"]
        self.assertEqual([row["revenue"] for row in quarters], [100.0, 150.0, 200.0, 250.0])
        self.assertEqual([row["n_income_attr_p"] for row in quarters], [10.0, 12.0, 13.0, 15.0])
        self.assertEqual(quarters[1]["evidence"]["revenue"][0]["field"], "revenue")

    def test_computes_cash_flow_quality_from_matching_cumulative_fields(self):
        report = analyze_fundamental_datasets(self.results, date(2025, 12, 31))
        latest = report["computed_metrics"]["cash_flow_quality"][-1]
        self.assertEqual(latest["end_date"], "20241231")
        self.assertAlmostEqual(latest["value"], 0.9)
        self.assertEqual({item["dataset"] for item in latest["evidence"]}, {"cashflow", "income"})

    def test_impairment_sign_is_not_interpreted_as_reversal(self):
        report = analyze_fundamental_datasets(self.results, date(2025, 12, 31))
        observation = report["impairment_observations"][0]
        self.assertEqual(observation["raw_value"], -2.0)
        self.assertEqual(observation["interpretation"], "provider_sign_only_no_reversal_inference")
        self.assertEqual(observation["evidence"]["field"], "assets_impair_loss")

    def test_verified_roe_and_roa_keep_field_level_evidence(self):
        report = analyze_fundamental_datasets(self.results, date(2025, 12, 31))
        indicators = report["verified_indicators"][-1]
        self.assertEqual(indicators["roe"], 12.0)
        self.assertEqual(indicators["evidence"]["roe"]["dataset"], "fina_indicator")
        self.assertEqual(indicators["evidence"]["roe"]["end_date"], "20241231")


if __name__ == "__main__":
    unittest.main()
