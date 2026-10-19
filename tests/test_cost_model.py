import unittest

from quant_platform.cost_model import build_cost_model, cost_breakdown


class CostModelTests(unittest.TestCase):
    def test_round_trip_rate_is_sum_of_components(self):
        model = build_cost_model("stock")
        expected = (
            model["buy_commission"]
            + model["sell_commission"]
            + model["stamp_tax"]
            + 2 * model["transfer_fee"]
            + 2 * model["slippage"]
        )
        self.assertAlmostEqual(model["round_trip_rate"], round(expected, 8))

    def test_etf_has_no_stamp_tax(self):
        self.assertEqual(build_cost_model("etf")["stamp_tax"], 0.0)

    def test_unknown_asset_falls_back_to_stock(self):
        self.assertEqual(build_cost_model("crypto")["asset_class"], "stock")

    def test_cost_breakdown_sums_to_total(self):
        model = build_cost_model("stock")
        breakdown = cost_breakdown(0.5, model)
        self.assertAlmostEqual(
            breakdown["total"],
            breakdown["buy_commission"]
            + breakdown["sell_commission"]
            + breakdown["stamp_tax"]
            + breakdown["transfer_fee"]
            + breakdown["slippage"],
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
