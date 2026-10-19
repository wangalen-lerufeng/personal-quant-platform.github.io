import unittest

import pandas as pd

from quant_platform.phase9_service import synthetic_etf_panel
from quant_platform.qlib_model import compute_group_returns, run_qlib_baseline, run_qlib_ensemble

SYMBOLS = ["510300.SH", "510500.SH", "159919.SZ", "512880.SH", "515790.SH"]


class QlibBaselineTests(unittest.TestCase):
    def test_lgb_model_produces_scores_and_candidates(self):
        panel = synthetic_etf_panel(SYMBOLS)
        result = run_qlib_baseline(
            panel, top_k=3, num_boost_round=40, num_leaves=15, learning_rate=0.05,
        )
        self.assertEqual(set(result["scores"].columns), {"timestamp", "symbol", "score"})
        self.assertFalse(result["scores"].empty)
        self.assertTrue(result["candidates"])
        self.assertTrue(all(len(symbols) <= 3 for symbols in result["candidates"].values()))


class QlibEnsembleTests(unittest.TestCase):
    def test_double_ensemble_produces_scores_and_candidates(self):
        panel = synthetic_etf_panel(SYMBOLS)
        result = run_qlib_ensemble(
            panel, top_k=3, num_models=3, num_boost_round=30, num_leaves=15, learning_rate=0.05,
        )
        self.assertEqual(set(result["scores"].columns), {"timestamp", "symbol", "score"})
        self.assertFalse(result["scores"].empty)
        self.assertTrue(result["candidates"])


class GroupReturnsTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for day in range(10):
            for symbol in range(5):
                score = float(symbol)
                rows.append({
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "symbol": f"S{symbol}",
                    "score": score,
                    "forward_return_1d": score * 0.01,
                })
        return pd.DataFrame(rows)

    def test_monotonic_ascending(self):
        panel = self._frame()
        scores = panel[["timestamp", "symbol", "score"]]
        result = compute_group_returns(scores, panel, "forward_return_1d", n_groups=5)
        self.assertTrue(result["available"])
        self.assertTrue(result["monotonic"])
        self.assertEqual(result["direction"], "ascending")
        self.assertGreater(result["long_short"], 0)
        self.assertEqual(set(result["groups"]), {"Q1", "Q2", "Q3", "Q4", "Q5"})

    def test_unavailable_on_empty(self):
        panel = self._frame()
        scores = panel[["timestamp", "symbol", "score"]].iloc[0:0]
        result = compute_group_returns(scores, panel, "forward_return_1d")
        self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
