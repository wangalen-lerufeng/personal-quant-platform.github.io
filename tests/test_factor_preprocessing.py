import unittest

import numpy as np
import pandas as pd

from quant_platform.factor_preprocessing import (
    cross_sectional_zscore,
    missing_report,
    preprocess_factor_panel,
    winsorize_mad,
)


def _frame():
    index = pd.MultiIndex.from_tuples(
        [("A", "20240102"), ("B", "20240102"), ("A", "20240103"), ("B", "20240103")],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]}, index=index)


class WinsorizeMadTests(unittest.TestCase):
    def test_clips_outliers(self):
        index = pd.MultiIndex.from_tuples([("A", "20240102")] * 5, names=["instrument", "datetime"])
        frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 100.0]}, index=index)
        result = winsorize_mad(frame, ["f1"])
        self.assertLess(result["f1"].max(), 100.0)


class CrossSectionalZscoreTests(unittest.TestCase):
    def test_each_day_zero_mean(self):
        frame = _frame()
        result = cross_sectional_zscore(frame, ["f1"])
        for _, group in result["f1"].groupby(level="datetime"):
            self.assertAlmostEqual(float(group.mean()), 0.0, places=6)


class PreprocessTests(unittest.TestCase):
    def test_keeps_shape_and_columns(self):
        frame = _frame()
        result = preprocess_factor_panel(frame, ["f1", "f2"])
        self.assertEqual(list(result.columns), ["f1", "f2"])
        self.assertEqual(len(result), len(frame))


class MissingReportTests(unittest.TestCase):
    def test_reports_missing_ratio(self):
        frame = _frame()
        frame.loc[("A", "20240102"), "f1"] = np.nan
        report = missing_report(frame, ["f1", "f2"])
        self.assertAlmostEqual(report["factor_missing_ratio"]["f1"], 0.25)
        self.assertAlmostEqual(report["factor_missing_ratio"]["f2"], 0.0)


if __name__ == "__main__":
    unittest.main()
