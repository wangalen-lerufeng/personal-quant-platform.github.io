import unittest

import pandas as pd

from quant_platform.double_ensemble import DoubleEnsembleConfig, run_double_ensemble


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
    return DoubleEnsembleConfig(
        n_models=5, n_estimators=20, num_leaves=7, min_child_samples=5, seed=7,
    )


class RunDoubleEnsembleTests(unittest.TestCase):
    def test_produces_scores_candidates_and_report(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        result = run_double_ensemble(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        self.assertEqual(set(result.scores.columns), {"timestamp", "symbol", "score"})
        self.assertTrue((result.scores["timestamp"] >= dates[40]).all())
        self.assertIn("horizons", result.factor_report)
        self.assertFalse(result.factor_report["future_data_used_for_feature"])
        self.assertEqual(result.metadata["model"], "double_ensemble")
        self.assertIn("feature_selection_history", result.metadata)
        self.assertIn("model_signature", result.metadata)

    def test_respects_chronological_split(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        result = run_double_ensemble(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        self.assertEqual(result.metadata["train_until"], pd.Timestamp(dates[40]).isoformat())
        self.assertTrue((result.scores["timestamp"] >= dates[40]).all())

    def test_is_reproducible_for_fixed_seed(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        first = run_double_ensemble(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        second = run_double_ensemble(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        pd.testing.assert_frame_equal(first.scores, second.scores)
        self.assertEqual(first.metadata["model_signature"], second.metadata["model_signature"])

    def test_candidates_match_top_k_by_score(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        result = run_double_ensemble(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        for timestamp, section in result.scores.groupby("timestamp"):
            expected = list(section.sort_values("score", ascending=False)["symbol"].head(2))
            self.assertEqual(result.candidates[pd.Timestamp(timestamp).isoformat()], expected)

    def test_feature_selection_reduces_features(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        result = run_double_ensemble(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        history = result.metadata["feature_selection_history"]
        self.assertEqual(len(history[0]), 5)
        self.assertLess(len(history[-1]), 5)

    def test_disabling_feature_selection_keeps_all_features(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        config = DoubleEnsembleConfig(
            n_models=4, n_estimators=20, num_leaves=7, min_child_samples=5,
            seed=7, feature_selection=False,
        )
        result = run_double_ensemble(
            panel, config=config, out_of_sample_start=dates[40], top_k=2,
        )
        history = result.metadata["feature_selection_history"]
        self.assertTrue(all(len(features) == 5 for features in history))

    def test_rejects_split_without_train_or_test(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        with self.assertRaisesRegex(ValueError, "train and test"):
            run_double_ensemble(panel, config=_small_config(), out_of_sample_start=dates[0])
        with self.assertRaisesRegex(ValueError, "train and test"):
            run_double_ensemble(panel, config=_small_config(), out_of_sample_start=dates[-1])

    def test_to_dict_marks_research_only(self):
        panel = _make_panel()
        dates = sorted(panel["timestamp"].unique())
        result = run_double_ensemble(
            panel, config=_small_config(), out_of_sample_start=dates[40], top_k=2,
        )
        payload = result.to_dict()
        self.assertFalse(payload["execution_enabled"])
        self.assertTrue(payload["research_only"])
        self.assertIn("scores", payload)
        self.assertIn("candidates", payload)


if __name__ == "__main__":
    unittest.main()
