import unittest

from quant_platform.capacity import estimate_capacity, square_root_impact


class CapacityTests(unittest.TestCase):
    def test_square_root_impact_scales_with_size(self):
        small = square_root_impact(1_000_000, 100_000_000, impact_coefficient=0.1)
        large = square_root_impact(10_000_000, 100_000_000, impact_coefficient=0.1)
        self.assertGreater(large, small)

    def test_estimate_capacity_reports_weakest_asset(self):
        volumes = {
            "A": [10_000_000, 12_000_000, 11_000_000],
            "B": [1_000_000, 1_200_000, 1_100_000],
        }
        report = estimate_capacity(volumes, participation_rate=0.01, target_notional=500_000)
        self.assertTrue(report["available"])
        self.assertEqual(report["weakest_asset"], "B")
        self.assertIn("A", report["per_asset"])
        self.assertIn("B", report["per_asset"])
        self.assertIsNotNone(report["per_asset"]["B"]["expected_fill_days"])
        self.assertIsNotNone(report["per_asset"]["B"]["impact_cost"])

    def test_estimate_capacity_empty(self):
        self.assertFalse(estimate_capacity({}, participation_rate=0.01)["available"])


if __name__ == "__main__":
    unittest.main()
