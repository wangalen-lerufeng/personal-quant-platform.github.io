import unittest

from quant_platform.phase9_service import synthetic_etf_panel
from quant_platform.qlib_model import run_qlib_baseline, run_qlib_ensemble

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


if __name__ == "__main__":
    unittest.main()
