import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_platform.factor_selection import select_factors
from quant_platform.qlib_alpha158 import (
    _effective_purge,
    _segment_boundaries,
    analyze_single_symbol_factors,
    run_alpha158,
)
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

    def test_run_alpha158_ensemble_handles_expression_feature_names(self):
        frame = _synthetic_frame(SYMBOLS, n_days=200)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            result = run_alpha158(
                SYMBOLS, "2024-01-01", "2024-10-01", qlib_dir=qlib_dir, top_k=3,
                model_type="ensemble", num_models=3, num_boost_round=20,
                num_leaves=15, learning_rate=0.05,
                feature_expressions=["Ref($close, 1)/$close - 1", "Mean($close, 20)/$close"],
            )
        self.assertFalse(result["scores"].empty)
        self.assertTrue(result["candidates"])

    def test_run_alpha158_supports_five_day_label(self):
        frame = _synthetic_frame(SYMBOLS, n_days=220)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            result = run_alpha158(
                SYMBOLS, "2024-01-01", "2024-11-01", qlib_dir=qlib_dir, top_k=3,
                num_boost_round=20, num_leaves=15, learning_rate=0.05,
                feature_expressions=["Ref($close, 1)/$close - 1", "Mean($close, 20)/$close"],
                label_horizon=5,
            )
        self.assertEqual(result["label_horizon"], 5)
        self.assertEqual(result["label_expression"], "Ref($close, -5)/$close - 1")
        self.assertFalse(result["scores"].empty)


class SegmentBoundariesTests(unittest.TestCase):
    def _frame(self, n_days=100):
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        index = pd.MultiIndex.from_product(
            [["A", "B"], dates], names=["instrument", "datetime"]
        )
        return pd.DataFrame({"x": 1.0}, index=index)

    def test_monthly_samples_do_not_treat_five_day_horizon_as_five_months(self):
        self.assertEqual(_effective_purge(5, 1, monthly_rebalance=True), 0)
        self.assertEqual(_effective_purge(5, 1, monthly_rebalance=False), 5)

    def test_single_split_has_purge_embargo_gaps(self):
        frame = self._frame()
        folds = _segment_boundaries(
            frame, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
            purge=1, embargo=1, walk_forward=False, n_walk_folds=3,
        )
        self.assertEqual(len(folds), 1)
        train_end, valid_start, valid_end, test_start, test_end = folds[0]
        self.assertLess(train_end, valid_start)
        self.assertLess(valid_end, test_start)
        self.assertLess(test_start, test_end)

    def test_walk_forward_returns_multiple_folds(self):
        frame = self._frame()
        folds = _segment_boundaries(
            frame, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
            purge=1, embargo=1, walk_forward=True, n_walk_folds=3,
        )
        self.assertEqual(len(folds), 3)
        for train_end, valid_start, valid_end, test_start, test_end in folds:
            self.assertLess(train_end, valid_start)
            self.assertLess(valid_end, test_start)


class SeedAveragingTests(unittest.TestCase):
    def test_multi_seed_returns_per_seed_scores(self):
        frame = _synthetic_frame(SYMBOLS, n_days=200)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            result = run_alpha158(
                SYMBOLS, "2024-01-01", "2024-10-01", qlib_dir=qlib_dir, top_k=3,
                num_boost_round=20, num_leaves=15, learning_rate=0.05,
                n_seeds=2,
            )
        self.assertEqual(len(result["seed_scores"]), 2)
        self.assertEqual(len(result["scores"]), len(result["seed_scores"][0]))
        self.assertEqual(set(result["scores"].columns), {"timestamp", "symbol", "score"})


class WalkForwardTests(unittest.TestCase):
    def test_walk_forward_covers_more_dates_than_single_split(self):
        frame = _synthetic_frame(SYMBOLS, n_days=200)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            single = run_alpha158(
                SYMBOLS, "2024-01-01", "2024-10-01", qlib_dir=qlib_dir, top_k=3,
                num_boost_round=20, num_leaves=15, learning_rate=0.05,
            )
            walk = run_alpha158(
                SYMBOLS, "2024-01-01", "2024-10-01", qlib_dir=qlib_dir, top_k=3,
                num_boost_round=20, num_leaves=15, learning_rate=0.05,
                walk_forward=True, n_walk_folds=3,
            )
        self.assertFalse(walk["scores"].empty)
        self.assertGreater(len(walk["scores"]), len(single["scores"]))


class SelectFactorsTests(unittest.TestCase):
    def _frame(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        rows = []
        for day_index, date in enumerate(dates):
            for symbol_index in range(5):
                # f0 强预测（高 IC），f1 与 f0 高度相关（冗余），f2 噪声
                f0 = float(day_index * 5 + symbol_index)
                f1 = f0 * 1.0 + 0.01 * symbol_index
                f2 = float((symbol_index * 7 + day_index) % 5)
                forward = float(symbol_index) * 0.01
                rows.append({
                    "datetime": date, "instrument": f"S{symbol_index}",
                    "f0": f0, "f1": f1, "f2": f2, "forward": forward,
                })
        frame = pd.DataFrame(rows).set_index(["datetime", "instrument"])
        return frame

    def test_select_factors_drops_redundant_and_noise(self):
        frame = self._frame()
        result = select_factors(
            frame, ["f0", "f1", "f2"],
            start=pd.Timestamp("2024-01-01"), train_end=pd.Timestamp("2024-12-31"),
            forward_column="forward", min_ic=0.02, min_icir=0.0, corr_threshold=0.9,
        )
        self.assertEqual(result["raw_count"], 3)
        self.assertIn("f0", result["selected"])
        self.assertNotIn("f1", result["selected"])  # 与 f0 冗余
        self.assertNotIn("f2", result["selected"])  # 噪声被 IC 阈值剔除
        self.assertEqual(result["total"], 3)
        self.assertIn("factors", result)


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
