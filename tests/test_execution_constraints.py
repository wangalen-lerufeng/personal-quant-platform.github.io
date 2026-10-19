import unittest

from quant_platform.execution_constraints import (
    apply_lot_size,
    is_limit_locked,
    limit_price_band,
    participation_notional_cap,
    t_plus_1_sellable,
)


class ExecutionConstraintsTests(unittest.TestCase):
    def test_apply_lot_size_rounds_down(self):
        self.assertEqual(apply_lot_size(250, 100), 200)
        self.assertEqual(apply_lot_size(99, 100), 0)
        self.assertEqual(apply_lot_size(250, 0), 250)

    def test_limit_price_band(self):
        lower, upper = limit_price_band(10.0, 0.1)
        self.assertAlmostEqual(lower, 9.0)
        self.assertAlmostEqual(upper, 11.0)

    def test_is_limit_locked(self):
        self.assertTrue(is_limit_locked("buy", 11.0, 10.0, 0.1))   # 涨停不能买
        self.assertTrue(is_limit_locked("sell", 9.0, 10.0, 0.1))   # 跌停不能卖
        self.assertFalse(is_limit_locked("buy", 10.5, 10.0, 0.1))
        self.assertFalse(is_limit_locked("sell", 10.5, 10.0, 0.1))

    def test_participation_notional_cap(self):
        self.assertEqual(participation_notional_cap(10_000_000, 0.01), 100_000.0)

    def test_t_plus_1_sellable(self):
        self.assertEqual(t_plus_1_sellable(1000, 400), 600)
        self.assertEqual(t_plus_1_sellable(300, 400), 0)


if __name__ == "__main__":
    unittest.main()
