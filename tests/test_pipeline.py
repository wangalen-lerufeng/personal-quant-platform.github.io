import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from quant_platform.backtest import BacktestEngine
from quant_platform.domain import AssetClass, MarketEvent
from quant_platform.pipeline import (
    TargetWeightStrategy,
    build_targets,
    plan_orders,
    submit_to_simulation,
)
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.simulation import PaperSimulationSession


def _decisions(mapping):
    return {symbol: SimpleNamespace(verdict=verdict) for symbol, verdict in mapping.items()}


class BuildTargetsTests(unittest.TestCase):
    def test_filters_to_pass_and_equal_weights(self):
        targets = build_targets(
            ["A", "B", "C"],
            _decisions({"A": "pass", "B": "pass", "C": "exclude"}),
        )
        self.assertEqual([t.symbol for t in targets], ["A", "B"])
        self.assertAlmostEqual(targets[0].weight, 0.5)
        self.assertAlmostEqual(targets[1].weight, 0.5)

    def test_applies_adjustment(self):
        targets = build_targets(
            ["A", "B"],
            _decisions({"A": "pass", "B": "pass"}),
            adjustment=0.8,
        )
        self.assertAlmostEqual(targets[0].weight, 0.4)

    def test_caps_total_weight_at_100_percent(self):
        targets = build_targets(
            ["A", "B"],
            _decisions({"A": "pass", "B": "pass"}),
            adjustment=1.2,
        )
        self.assertAlmostEqual(targets[0].weight, 0.5)
        self.assertAlmostEqual(targets[1].weight, 0.5)

    def test_empty_when_no_pass(self):
        self.assertEqual(build_targets(["A"], _decisions({"A": "observe"})), [])


class PlanOrdersTests(unittest.TestCase):
    def test_generates_buys_and_sells(self):
        targets = [SimpleNamespace(symbol="A", weight=0.5), SimpleNamespace(symbol="B", weight=0.5)]
        plan = plan_orders({"A": 0.2}, targets, RiskEngine(RiskLimits()))
        self.assertEqual(len(plan.orders), 2)
        sides = {order.symbol: order.side.value for order in plan.orders}
        self.assertEqual(sides, {"A": "buy", "B": "buy"})
        self.assertEqual(plan.orders[0].quantity, 0.3)

    def test_rejects_over_limit_orders(self):
        targets = [SimpleNamespace(symbol="B", weight=0.5)]
        plan = plan_orders({}, targets, RiskEngine(RiskLimits(max_order_quantity=0.4)))
        self.assertEqual(plan.orders, ())
        self.assertEqual(plan.rejected[0]["symbol"], "B")
        self.assertEqual(plan.rejected[0]["reason"], "order_quantity_exceeded")

    def test_respects_kill_switch(self):
        targets = [SimpleNamespace(symbol="A", weight=0.5)]
        plan = plan_orders({}, targets, RiskEngine(RiskLimits(kill_switch=True)))
        self.assertEqual(plan.orders, ())
        self.assertEqual(plan.rejected[0]["reason"], "kill_switch_active")

    def test_audit_records_risk_gate(self):
        targets = [SimpleNamespace(symbol="A", weight=0.5)]
        plan = plan_orders({}, targets, RiskEngine(RiskLimits()))
        stages = [entry.stage for entry in plan.audit]
        self.assertIn("risk_gate", stages)


class SubmitToSimulationTests(unittest.TestCase):
    def test_submits_accepted_orders(self):
        targets = [SimpleNamespace(symbol="A", weight=0.5)]
        plan = plan_orders({}, targets, RiskEngine(RiskLimits()))
        session = PaperSimulationSession("paper-main", initial_cash=100000.0)
        result = submit_to_simulation(plan, session)
        self.assertEqual(result["order_count"], 1)
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["paper_only"])
        self.assertEqual(result["submitted"][0]["symbol"], "A")


class TargetWeightStrategyTests(unittest.TestCase):
    def _event(self, symbol, day, price=10.0):
        return MarketEvent(symbol, AssetClass.ETF, price, datetime(2026, 1, day, tzinfo=timezone.utc))

    def test_backtest_reaches_target_symbols(self):
        strategy = TargetWeightStrategy({"A": 1.0, "B": 1.0})
        events = [self._event("A", 1), self._event("B", 1), self._event("A", 2), self._event("B", 2)]
        result = BacktestEngine(strategy).run(events)
        self.assertEqual(result.signal_count, 2)
        self.assertEqual(result.trade_count, 2)
        self.assertEqual({trade["symbol"] for trade in result.trades}, {"A", "B"})


class ExecutionPlanToDictTests(unittest.TestCase):
    def test_to_dict_marks_research_only(self):
        targets = [SimpleNamespace(symbol="A", weight=0.5)]
        plan = plan_orders({}, targets, RiskEngine(RiskLimits()))
        payload = plan.to_dict()
        self.assertFalse(payload["execution_enabled"])
        self.assertTrue(payload["research_only"])
        self.assertIn("orders", payload)
        self.assertIn("audit", payload)


if __name__ == "__main__":
    unittest.main()
