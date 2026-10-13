import unittest

from quant_platform.domain import AssetClass, OrderIntent, OrderSide
from quant_platform.market_rules import MarketRuleBook
from quant_platform.risk import RiskEngine, RiskLimits


class RiskTests(unittest.TestCase):
    def test_risk_engine_rejects_order_when_kill_switch_is_active(self):
        engine = RiskEngine(RiskLimits(kill_switch=True))
        intent = OrderIntent(symbol="EURUSD", side=OrderSide.BUY, quantity=0.01, strategy_id="demo")
        decision = engine.evaluate(intent)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "kill_switch_active")

    def test_risk_engine_allows_small_order_by_default(self):
        engine = RiskEngine(RiskLimits())
        intent = OrderIntent(symbol="510300", side=OrderSide.BUY, quantity=100, strategy_id="demo")
        self.assertTrue(engine.evaluate(intent).allowed)

    def test_risk_engine_rejects_invalid_market_lot(self):
        engine = RiskEngine(RiskLimits(), MarketRuleBook())
        intent = OrderIntent(symbol="510300", side=OrderSide.BUY, quantity=50, strategy_id="demo", asset_class=AssetClass.ETF)

        decision = engine.evaluate(intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "market_rule_violation")

    def test_risk_engine_rejects_order_above_notional_limit(self):
        engine = RiskEngine(RiskLimits(max_order_value=10_000))
        intent = OrderIntent(
            symbol="510300",
            side=OrderSide.BUY,
            quantity=1_000,
            strategy_id="demo",
            asset_class=AssetClass.ETF,
            reference_price=12.0,
        )

        decision = engine.evaluate(intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "order_value_exceeded")

    def test_risk_engine_requires_reference_price_for_notional_limit(self):
        engine = RiskEngine(RiskLimits(max_order_value=10_000))
        intent = OrderIntent("510300", OrderSide.BUY, 100, "demo", asset_class=AssetClass.ETF)

        decision = engine.evaluate(intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "reference_price_required")

    def test_risk_engine_rejects_order_above_position_limit(self):
        engine = RiskEngine(RiskLimits(max_position_quantity=100))
        engine.record_fill("510300", OrderSide.BUY, 80)
        intent = OrderIntent("510300", OrderSide.BUY, 30, "demo", asset_class=AssetClass.ETF)

        decision = engine.evaluate(intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "position_quantity_exceeded")
