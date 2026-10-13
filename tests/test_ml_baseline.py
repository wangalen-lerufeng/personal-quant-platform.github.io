import unittest

import pandas as pd

from quant_platform.ml_baseline import (
    DEFAULT_FEATURE_COLUMNS,
    BaselineConfig,
    compute_cross_sectional_features,
    run_baseline,
)


def _make_panel(n_symbols=4, n_days=60):
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rows = []
    for si, symbol in enumerate(symbols):
        base = 10.0 * (si + 1)
        growth = 0.005 * (si + 1)
        closes = [base * (1 + growth * di) for di in range(n_days)]
        for di, ts in enumerate(dates):
            forward = closes[di + 5] / closes[di] - 1 if di + 5 < n_days else None
            rows.append({
                "timestamp": ts,
                "symbol": symbol,
                "adj_close": closes[di],
                "volume": 1000 * (si + 1) + 10 * di,
                "amount": 1_000_000 * (si + 1) + 1000 * di,
                "forward_return_5d": forward,
            })
    return pd.DataFrame(rows)


def _small_config():
    return BaselineConfig(n_estimators=20, num_leaves=7, min_child_samples=5, seed=7)


class ComputeFeaturesTests(unittest.TestCase):
    def test_features_are_present_and_causal(self):
        panel = _make_panel(n_symbols=3, n_days=30)
        frame = compute_cross_sectional_features(panel)
        for column in DEFAULT_FEATURE_COLUMNS:
            self.assertIn(column, frame.columns)
        first_symbol = frame[frame["symbol"] == "S0"].sort_values("timestamp").reset_index(drop=True)
        self.assertTrue(pd.isna(first_symbol["return_1d"].iloc[0]))
        self.assertTrue(first_symbol["momentum_20d"].iloc[:20].isna().all())
        self.assertTrue(first_symbol["volatility_20d"].iloc[:19].isna().all())

    def test_momentum_matches_close_ratio(self):
        panel = _make_panel(n_symbols=2, n_days=30)
        frame = compute_cross_sectional_features(panel)
        s0 = frame[frame["symbol"] == "S0"].sort_values("timestamp").reset_index(drop=True)
        expected = s0["adj_close"].iloc[20] / s0["adj_close"].iloc[0] - 1
        self.assertAlmostEqual(s0["momentum_20d"].iloc[20], expected)


class RunBaselineTests(unittest.TestCase):
    def test_produces_scores_candidates_and_report(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        oos = dates[40]
        result = run_baseline(
            panel, config=_small_config(), out_of_sample_start=oos, top_k=2,
        )
        self.assertEqual(set(result.scores.columns), {"timestamp", "symbol", "score"})
        self.assertTrue((result.scores["timestamp"] >= oos).all())
        for date, symbols in result.candidates.items():
            self.assertLessEqual(len(symbols), 2)
        self.assertIn("horizons", result.factor_report)
        self.assertIn("observations", result.factor_report)
        self.assertEqual(result.metadata["train_until"], pd.Timestamp(oos).isoformat())
        self.assertIn("feature_columns", result.metadata)
        self.assertIn("model_signature", result.metadata)

    def test_respects_chronological_split(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        oos = dates[40]
        result = run_baseline(
            panel, config=_small_config(), out_of_sample_start=oos, top_k=2,
        )
        self.assertEqual(result.metadata["train_until"], pd.Timestamp(oos).isoformat())
        self.assertTrue((result.scores["timestamp"] >= oos).all())

    def test_is_reproducible_for_fixed_seed(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        oos = dates[40]
        first = run_baseline(
            panel, config=_small_config(), out_of_sample_start=oos, top_k=2,
        )
        second = run_baseline(
            panel, config=_small_config(), out_of_sample_start=oos, top_k=2,
        )
        pd.testing.assert_frame_equal(first.scores, second.scores)
        self.assertEqual(first.metadata["model_signature"], second.metadata["model_signature"])

    def test_candidates_match_top_k_by_score(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        oos = dates[40]
        result = run_baseline(
            panel, config=_small_config(), out_of_sample_start=oos, top_k=2,
        )
        for timestamp, section in result.scores.groupby("timestamp"):
            expected = list(section.sort_values("score", ascending=False)["symbol"].head(2))
            key = pd.Timestamp(timestamp).isoformat()
            self.assertEqual(result.candidates[key], expected)

    def test_rejects_split_without_train_or_test(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        with self.assertRaisesRegex(ValueError, "train and test"):
            run_baseline(panel, config=_small_config(), out_of_sample_start=dates[0])
        with self.assertRaisesRegex(ValueError, "train and test"):
            run_baseline(panel, config=_small_config(), out_of_sample_start=dates[-1])

    def test_to_dict_marks_research_only(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        result = run_baseline(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        payload = result.to_dict()
        self.assertFalse(payload["execution_enabled"])
        self.assertTrue(payload["research_only"])
        self.assertIn("scores", payload)
        self.assertIn("candidates", payload)


if __name__ == "__main__":
    unittest.main()
