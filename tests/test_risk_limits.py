import unittest

from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.risk import RiskEngine, RiskLimits


class RiskLimitStateTests(unittest.TestCase):
    def test_daily_loss_limit_rejects_new_order(self):
        engine = RiskEngine(RiskLimits(max_daily_loss=500))
        engine.record_pnl(-500)
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo")

        decision = engine.evaluate(intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "daily_loss_limit")

    def test_drawdown_limit_rejects_new_order(self):
        engine = RiskEngine(RiskLimits(max_drawdown=1000))
        engine.record_drawdown(1000)
        intent = OrderIntent("510300", OrderSide.BUY, 100, "demo")

        decision = engine.evaluate(intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "drawdown_limit")

    def test_limits_can_be_loaded_from_mapping(self):
        limits = RiskLimits.from_mapping(
            {
                "kill_switch": "false",
                "max_order_quantity": 100,
                "max_position_quantity": 200,
                "max_order_value": 10_000,
                "max_daily_loss": 500,
                "max_drawdown": 1_000,
            }
        )

        self.assertFalse(limits.kill_switch)
        self.assertEqual(limits.max_order_quantity, 100)
        self.assertEqual(limits.max_position_quantity, 200)
        self.assertEqual(limits.max_order_value, 10_000)
        self.assertEqual(limits.max_daily_loss, 500)
        self.assertEqual(limits.max_drawdown, 1_000)

    def test_nested_example_risk_configuration_is_supported(self):
        limits = RiskLimits.from_mapping(
            {"risk": {"max_order_quantity": 100, "max_daily_loss": 500, "kill_switch": False}}
        )

        self.assertEqual(limits.max_order_quantity, 100)
        self.assertEqual(limits.max_daily_loss, 500)
        self.assertFalse(limits.kill_switch)

    def test_invalid_nested_risk_section_is_rejected(self):
        with self.assertRaises(ValueError):
            RiskLimits.from_mapping({"risk": "not-a-mapping"})

    def test_limits_reject_non_positive_quantity_or_negative_thresholds(self):
        with self.assertRaises(ValueError):
            RiskLimits(max_order_quantity=0)
        with self.assertRaises(ValueError):
            RiskLimits(max_order_value=-1)
        with self.assertRaises(ValueError):
            RiskLimits(max_daily_loss=-1)
