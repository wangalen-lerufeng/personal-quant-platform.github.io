from datetime import datetime, timezone
import unittest

from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.execution import ExecutionGateway, InMemoryExecutionGateway, OrderLifecycle, OrderState, PaperMatchingEngine


class ExecutionTests(unittest.TestCase):
    def test_paper_gateway_matches_execution_gateway_protocol(self):
        gateway = InMemoryExecutionGateway()

        self.assertIsInstance(gateway, ExecutionGateway)
    def test_submit_order_creates_pending_order_and_audit_event(self):
        gateway = InMemoryExecutionGateway()
        intent = OrderIntent(symbol="EURUSD", side=OrderSide.BUY, quantity=0.01, strategy_id="demo")

        receipt = gateway.submit_order(intent)

        self.assertEqual(receipt.status, OrderState.PENDING.value)
        self.assertEqual(len(gateway.audit_events), 1)
        self.assertEqual(gateway.audit_events[0].event_type, "order_submitted")

    def test_order_lifecycle_allows_pending_to_filled(self):
        lifecycle = OrderLifecycle(order_id="order-1")

        lifecycle.transition(OrderState.ACCEPTED)
        lifecycle.transition(OrderState.FILLED)

        self.assertEqual(lifecycle.state, OrderState.FILLED)

    def test_order_lifecycle_rejects_invalid_transition(self):
        lifecycle = OrderLifecycle(order_id="order-1")

        with self.assertRaises(ValueError):
            lifecycle.transition(OrderState.FILLED)

    def test_gateway_can_cancel_pending_order(self):
        gateway = InMemoryExecutionGateway()
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo")
        receipt = gateway.submit_order(intent)

        state = gateway.cancel_order(receipt.order_id)

        self.assertEqual(state, OrderState.CANCELED)
        self.assertEqual(gateway.orders[receipt.order_id].state, OrderState.CANCELED)
        self.assertEqual(gateway.audit_events[-1].event_type, "order_canceled")

    def test_gateway_exposes_empty_positions_and_health(self):
        gateway = InMemoryExecutionGateway()

        self.assertEqual(gateway.get_positions(), {})
        self.assertEqual(gateway.health_check()["status"], "ready")
        self.assertFalse(gateway.health_check()["execution_enabled"])

    def test_gateway_can_advance_order_and_emit_fill_audit(self):
        gateway = InMemoryExecutionGateway()
        receipt = gateway.submit_order(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))

        gateway.transition_order(receipt.order_id, OrderState.ACCEPTED)
        state = gateway.transition_order(receipt.order_id, OrderState.FILLED)

        self.assertEqual(state, OrderState.FILLED)
        self.assertEqual(gateway.audit_events[-1].event_type, "order_filled")

    def test_filled_orders_update_positions(self):
        gateway = InMemoryExecutionGateway()
        buy = gateway.submit_order(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))
        gateway.transition_order(buy.order_id, OrderState.ACCEPTED)
        gateway.transition_order(buy.order_id, OrderState.FILLED)
        sell = gateway.submit_order(OrderIntent("EURUSD", OrderSide.SELL, 0.004, "demo"))
        gateway.transition_order(sell.order_id, OrderState.ACCEPTED)
        gateway.transition_order(sell.order_id, OrderState.FILLED)

        self.assertAlmostEqual(gateway.get_positions()["EURUSD"], 0.006)

    def test_gateway_rejects_unknown_order_transition(self):
        gateway = InMemoryExecutionGateway()

        with self.assertRaises(KeyError):
            gateway.transition_order("missing", OrderState.ACCEPTED)

    def test_gateway_tracks_partial_fills_and_positions(self):
        gateway = InMemoryExecutionGateway()
        receipt = gateway.submit_order(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))
        gateway.transition_order(receipt.order_id, OrderState.ACCEPTED)

        state = gateway.record_fill(receipt.order_id, 0.004)
        self.assertEqual(state, OrderState.PARTIALLY_FILLED)
        self.assertAlmostEqual(gateway.get_positions()["EURUSD"], 0.004)
        state = gateway.record_fill(receipt.order_id, 0.006)
        self.assertEqual(state, OrderState.FILLED)
        self.assertAlmostEqual(gateway.get_positions()["EURUSD"], 0.01)

    def test_matching_engine_applies_latency_and_partial_fill(self):
        engine = PaperMatchingEngine(initial_cash=1000, latency_events=1, max_participation_rate=0.1)
        order_id = engine.submit(OrderIntent("510300.SH", OrderSide.BUY, 100, "cta"))

        self.assertEqual(engine.on_market("510300.SH", price=4.0, volume=500), [])
        fills = engine.on_market("510300.SH", price=4.0, volume=500)

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 50)
        self.assertEqual(engine.orders[order_id].state, OrderState.PARTIALLY_FILLED)

    def test_matching_engine_caps_buy_by_available_cash(self):
        engine = PaperMatchingEngine(initial_cash=100, latency_events=0, max_participation_rate=1.0)
        order_id = engine.submit(OrderIntent("510300.SH", OrderSide.BUY, 100, "cta"))

        fills = engine.on_market("510300.SH", price=4.0, volume=1000)

        self.assertEqual(fills[0].quantity, 25)
        self.assertEqual(engine.cash, 0)
        self.assertEqual(engine.orders[order_id].state, OrderState.PARTIALLY_FILLED)
