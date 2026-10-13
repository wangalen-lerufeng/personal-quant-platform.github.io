import unittest

from quant_platform.order_features import analyze_order_flow


class OrderFeatureTests(unittest.TestCase):
    def test_order_flow_summary_is_deterministic(self):
        trades = [
            {"side": "buy", "price": 100.0, "quantity": 10},
            {"side": "sell", "price": 101.0, "quantity": 5},
            {"side": "buy", "price": 102.0, "quantity": 20},
        ]

        result = analyze_order_flow(trades, large_trade_threshold=15)

        self.assertEqual(result["buy_volume"], 30)
        self.assertEqual(result["sell_volume"], 5)
        self.assertAlmostEqual(result["volume_imbalance"], 25 / 35)
        self.assertAlmostEqual(result["large_trade_ratio"], 20 / 35)
        self.assertAlmostEqual(result["price_impact"], 0.02)

    def test_order_flow_rejects_invalid_side(self):
        with self.assertRaises(ValueError):
            analyze_order_flow([{"side": "unknown", "price": 1, "quantity": 1}])

