import unittest

from quant_platform.domain import AssetClass, OrderIntent, OrderSide
from quant_platform.market_rules import MarketRuleBook, RuleViolation


class MarketRulesTests(unittest.TestCase):
    def test_etf_requires_100_share_lot(self):
        rules = MarketRuleBook()
        intent = OrderIntent("510300", OrderSide.BUY, 50, "demo")

        with self.assertRaises(RuleViolation):
            rules.validate(AssetClass.ETF, intent)

    def test_mt5_fx_accepts_fractional_lot_quantity(self):
        rules = MarketRuleBook()
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo")

        self.assertIsNone(rules.validate(AssetClass.FX, intent))

    def test_fx_valid_decimal_step_is_not_rejected_by_float_rounding(self):
        rules = MarketRuleBook()
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.03, "demo")

        self.assertIsNone(rules.validate(AssetClass.FX, intent))
