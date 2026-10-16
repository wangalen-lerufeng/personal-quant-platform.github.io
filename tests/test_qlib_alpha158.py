import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_platform.qlib_alpha158 import analyze_single_symbol_factors, compute_factor_ic_report, run_alpha158
from quant_platform.qlib_dump import dump_etf_to_qlib

SYMBOLS = ["510300.SH", "510500.SH", "159919.SZ", "512880.SH", "515790.SH"]


def _synthetic_frame(symbols, n_days=200):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for index, symbol in enumerate(symbols):
        for day_index, date in enumerate(dates):
            close = 4.0 * (index + 1) * (1 + 0.002 * day_index + 0.01 * np.sin(day_index / 5 + index))
            rows.append({
                "symbol": symbol,
                "date": date.strftime("%Y%m%d"),
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "vwap": close,
                "volume": 1000.0,
                "amount": 1e6,
            })
    return pd.DataFrame(rows)


class Alpha158Tests(unittest.TestCase):
    def test_run_alpha158_produces_scores_and_candidates(self):
        frame = _synthetic_frame(SYMBOLS, n_days=200)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            result = run_alpha158(
                SYMBOLS, "2024-01-01", "2024-10-01", qlib_dir=qlib_dir, top_k=3,
                num_boost_round=30, num_leaves=15, learning_rate=0.05,
            )
        self.assertEqual(set(result["scores"].columns), {"timestamp", "symbol", "score"})
        self.assertFalse(result["scores"].empty)
        self.assertTrue(result["candidates"])
        self.assertTrue(all(len(symbols) <= 3 for symbols in result["candidates"].values()))

    def test_run_alpha158_with_custom_factor_expressions(self):
        frame = _synthetic_frame(SYMBOLS, n_days=200)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            result = run_alpha158(
                SYMBOLS, "2024-01-01", "2024-10-01", qlib_dir=qlib_dir, top_k=3,
                num_boost_round=20, num_leaves=15, learning_rate=0.05,
                feature_expressions=["Ref($close, 1)/$close - 1", "Mean($close, 20)/$close"],
            )
        self.assertFalse(result["scores"].empty)
        self.assertTrue(result["candidates"])


class FactorICReportTests(unittest.TestCase):
    def test_compute_factor_ic_report(self):
        frame = _synthetic_frame(SYMBOLS, n_days=100)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            report = compute_factor_ic_report(
                SYMBOLS, "2024-01-01", "2024-10-01", qlib_dir=qlib_dir,
                feature_expressions=["Ref($close,1)/$close-1", "Mean($close,20)/$close"],
            )
        self.assertEqual(report["total"], 2)
        self.assertTrue(report["factors"])
        self.assertFalse(report["execution_enabled"])
        for factor in report["factors"]:
            self.assertIn("rank_ic", factor)
            self.assertIn("icir", factor)
            self.assertIn("days", factor)


class SingleSymbolFactorTests(unittest.TestCase):
    def test_analyze_single_symbol_factors(self):
        symbol = "09660.HK"
        frame = _synthetic_frame([symbol], n_days=100)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("quant_platform.qlib_dump.build_ohlcv_panel", return_value=frame):
                result = analyze_single_symbol_factors(
                    symbol, ["Ref($close,20)/$close-1", "Std($close,20)/$close"],
                    "2024-01-01", "2024-10-01",
                    qlib_dir=Path(tmp) / "qlib_factor", asset_class="hk",
                )
        self.assertEqual(result["symbol"], symbol)
        self.assertEqual(len(result["factors"]), 2)
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["research_only"])
        for factor in result["factors"]:
            self.assertIn("mean", factor)
            self.assertIn("latest", factor)
            self.assertIn("percentile", factor)
            self.assertIn("ic", factor)
            self.assertTrue(factor["series"])


if __name__ == "__main__":
    unittest.main()
