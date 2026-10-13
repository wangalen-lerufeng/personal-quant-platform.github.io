import unittest

from quant_platform.domain import OrderSide
from quant_platform.portfolio import PositionBook


class PortfolioTests(unittest.TestCase):
    def test_buy_increases_position(self):
        book = PositionBook()

        book.apply_fill("EURUSD", OrderSide.BUY, 0.01, 1.08)

        self.assertEqual(book.quantity("EURUSD"), 0.01)

    def test_sell_reduces_position(self):
        book = PositionBook()
        book.apply_fill("EURUSD", OrderSide.BUY, 0.02, 1.08)
        book.apply_fill("EURUSD", OrderSide.SELL, 0.01, 1.081)

        self.assertEqual(book.quantity("EURUSD"), 0.01)

    def test_snapshot_marks_position_to_market(self):
        book = PositionBook()
        book.apply_fill("EURUSD", OrderSide.BUY, 10, 1.0)
        book.apply_fill("EURUSD", OrderSide.BUY, 10, 1.2)

        snapshot = book.snapshot({"EURUSD": 1.3})

        self.assertEqual(snapshot["positions"]["EURUSD"]["quantity"], 20)
        self.assertAlmostEqual(snapshot["positions"]["EURUSD"]["average_price"], 1.1)
        self.assertAlmostEqual(snapshot["positions"]["EURUSD"]["unrealized_pnl"], 4.0)
        self.assertAlmostEqual(snapshot["equity"], 26.0)

    def test_snapshot_includes_realized_and_total_pnl_after_sell(self):
        book = PositionBook()
        book.apply_fill("EURUSD", OrderSide.BUY, 10, 1.0)
        book.apply_fill("EURUSD", OrderSide.SELL, 4, 1.2)

        snapshot = book.snapshot({"EURUSD": 1.1})

        self.assertAlmostEqual(snapshot["realized_pnl"], 0.8)
        self.assertAlmostEqual(snapshot["unrealized_pnl"], 0.6)
        self.assertAlmostEqual(snapshot["total_pnl"], 1.4)
