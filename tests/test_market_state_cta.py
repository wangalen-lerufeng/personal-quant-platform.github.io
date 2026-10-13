import unittest

import pandas as pd

from quant_platform.market_state import classify_market_state
from quant_platform.cta import donchian_signal


class MarketStateCtaTests(unittest.TestCase):
    def test_market_state_is_explainable(self):
        frame = pd.DataFrame({"close": [10, 10.2, 10.4, 10.6, 10.8], "high": [10.1, 10.3, 10.5, 10.7, 10.9], "low": [9.9, 10.1, 10.3, 10.5, 10.7]})
        result = classify_market_state(frame, window=3)
        self.assertIn(result["state"], {"trend", "range", "high_volatility", "insufficient_data"})
        self.assertTrue(result["evidence"])

    def test_donchian_signal_uses_shifted_breakout(self):
        frame = pd.DataFrame({"high": [10, 11, 12, 13], "low": [8, 9, 10, 11], "close": [9, 10, 11, 14]})
        result = donchian_signal(frame, entry_window=2, exit_window=2)
        self.assertEqual(result.loc[2, "signal"], 0)
        self.assertEqual(result.loc[3, "signal"], 1)

