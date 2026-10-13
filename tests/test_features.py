import unittest

import pandas as pd

from quant_platform.features import add_basic_features


class FeatureTests(unittest.TestCase):
    def test_features_are_lagged_and_do_not_use_future_close(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, tz="UTC"),
            "open": [10, 11, 12, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10, 11, 12, 13, 14],
            "volume": [100, 110, 120, 130, 140],
        })
        result = add_basic_features(frame, window=2)
        self.assertTrue(pd.isna(result.loc[0, "return_1d"]))
        self.assertAlmostEqual(result.loc[2, "momentum_2d"], 0.2)
        self.assertAlmostEqual(result.loc[2, "return_1d"], 1 / 11)
        self.assertTrue(pd.isna(result.loc[0, "atr_2d"]))
        self.assertIn("ma_distance_2d", result)

