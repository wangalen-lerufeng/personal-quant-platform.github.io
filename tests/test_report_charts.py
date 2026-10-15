import unittest

import numpy as np
import pandas as pd

from quant_platform.report_charts import compute_report_charts


def _panel(n_days=60, n_symbols=5):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for s_i, symbol in enumerate([f"S{i}" for i in range(n_symbols)]):
        close = 10.0 * (s_i + 1)
        prev = None
        for d_i, date in enumerate(dates):
            close = close * (1 + 0.001 * (s_i + 1) + 0.002 * np.sin(d_i / 4 + s_i))
            ret = None
            if prev is not None:
                ret = close / prev - 1
            rows.append({"timestamp": date, "symbol": symbol, "forward_return_1d": ret})
            prev = close
    return pd.DataFrame(rows)


def _scores_and_candidates(panel, n_symbols=5, top_k=2):
    dates = sorted(panel["timestamp"].unique())
    score_rows = []
    candidates = {}
    for d_i, date in enumerate(dates[:-1]):
        for s_i in range(n_symbols):
            score_rows.append({"timestamp": date, "symbol": f"S{s_i}", "score": float((d_i + s_i) % n_symbols)})
        top = [f"S{s}" for s in sorted(range(n_symbols), key=lambda s: (d_i + s) % n_symbols)[:top_k]]
        candidates[date.isoformat()] = top
    return pd.DataFrame(score_rows), candidates


class ReportChartsTests(unittest.TestCase):
    def test_computes_series_and_metrics(self):
        panel = _panel()
        scores, candidates = _scores_and_candidates(panel)
        charts = compute_report_charts(scores, panel, candidates, label_column="forward_return_1d")
        self.assertIsNotNone(charts)
        self.assertGreater(len(charts["dates"]), 0)
        self.assertEqual(len(charts["dates"]), len(charts["strategy_nav"]))
        self.assertEqual(len(charts["dates"]), len(charts["benchmark_nav"]))
        self.assertEqual(len(charts["dates"]), len(charts["excess"]))
        self.assertEqual(len(charts["dates"]), len(charts["drawdown"]))
        self.assertEqual(len(charts["dates"]), len(charts["turnover"]))
        self.assertEqual(len(charts["dates"]), len(charts["ic_series"]))
        self.assertIn("information_ratio", charts["metrics"])
        self.assertIn("max_drawdown", charts["metrics"])
        self.assertIn("sharpe", charts["metrics"])
        self.assertIn("mean_ic", charts["metrics"])
        # NAVs are strictly positive cumulative products
        self.assertTrue(all(v > 0 for v in charts["strategy_nav"]))
        self.assertTrue(all(v > 0 for v in charts["benchmark_nav"]))

    def test_returns_none_for_empty_input(self):
        self.assertIsNone(compute_report_charts(pd.DataFrame(), pd.DataFrame(), {}))
        self.assertIsNone(compute_report_charts(None, _panel(), {"2024-01-02": ["S0"]}))

    def test_drawdown_is_never_positive(self):
        panel = _panel()
        scores, candidates = _scores_and_candidates(panel)
        charts = compute_report_charts(scores, panel, candidates, label_column="forward_return_1d")
        self.assertLessEqual(max(charts["drawdown"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
