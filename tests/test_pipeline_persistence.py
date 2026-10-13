import unittest

from quant_platform.adapters import Mt5PaperGateway
from quant_platform.adapters import QmtAdapter
from quant_platform.domain import AssetClass
from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.storage import SqliteAuditStore
from quant_platform.trading import TradingPipeline
from quant_platform.execution import OrderState
from quant_platform.control import ControlPlane
from quant_platform.config import RuntimeConfig
from quant_platform.portfolio import PositionBook


class PipelinePersistenceTests(unittest.TestCase):
    def test_approved_order_is_persisted_with_gateway_order_id(self):
        store = SqliteAuditStore(":memory:")
        pipeline = TradingPipeline(
            RiskEngine(RiskLimits()),
            Mt5PaperGateway(),
            audit_store=store,
        )
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo")

        result = pipeline.submit(intent)

        self.assertTrue(result.allowed)
        self.assertEqual(len(store.list_orders()), 1)
        self.assertEqual(store.list_orders()[0]["status"], "accepted")

    def test_rejected_order_is_not_persisted(self):
        store = SqliteAuditStore(":memory:")
        pipeline = TradingPipeline(
            RiskEngine(RiskLimits(kill_switch=True)),
            Mt5PaperGateway(),
            audit_store=store,
        )
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo")

        pipeline.submit(intent)

        self.assertEqual(store.list_orders(), [])

    def test_execution_rejection_is_persisted_for_audit(self):
        store = SqliteAuditStore(":memory:")
        pipeline = TradingPipeline(RiskEngine(RiskLimits()), QmtAdapter(), audit_store=store)
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo", asset_class=AssetClass.FX)

        result = pipeline.submit(intent)

        self.assertFalse(result.allowed)
        orders = store.list_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["status"], "rejected")

    def test_execution_status_update_is_persisted(self):
        store = SqliteAuditStore(":memory:")
        pipeline = TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway(), audit_store=store)
        result = pipeline.submit(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))

        pipeline.update_execution_status(result.receipt.order_id, OrderState.FILLED)

        self.assertEqual(store.list_orders()[0]["status"], "filled")

    def test_invalid_execution_status_is_rejected(self):
        store = SqliteAuditStore(":memory:")
        pipeline = TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway(), audit_store=store)
        result = pipeline.submit(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))

        with self.assertRaises(ValueError):
            pipeline.update_execution_status(result.receipt.order_id, "unknown")

    def test_fill_can_update_control_plane_position_book(self):
        store = SqliteAuditStore(":memory:")
        control = ControlPlane(
            RuntimeConfig(), RiskEngine(RiskLimits()), store, position_book=PositionBook()
        )
        pipeline = TradingPipeline(
            RiskEngine(RiskLimits()), Mt5PaperGateway(), fill_handler=control.record_fill
        )
        result = pipeline.submit(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))

        pipeline.record_fill(result.receipt.order_id, 1.08)

        self.assertAlmostEqual(control.portfolio_state({"EURUSD": 1.09})["market_value"], 0.0109)

    def test_partial_fills_cannot_exceed_submitted_quantity(self):
        handler_calls = []
        pipeline = TradingPipeline(
            RiskEngine(RiskLimits()),
            Mt5PaperGateway(),
            fill_handler=lambda *args: handler_calls.append(args),
        )
        result = pipeline.submit(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))

        pipeline.record_fill(result.receipt.order_id, 1.08, 0.006)
        with self.assertRaises(ValueError):
            pipeline.record_fill(result.receipt.order_id, 1.08, 0.005)

        self.assertEqual(len(handler_calls), 1)

    def test_fill_updates_audit_status_for_partial_and_full_completion(self):
        store = SqliteAuditStore(":memory:")
        pipeline = TradingPipeline(
            RiskEngine(RiskLimits()),
            Mt5PaperGateway(),
            audit_store=store,
            fill_handler=lambda *_args: None,
        )
        result = pipeline.submit(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"))

        pipeline.record_fill(result.receipt.order_id, 1.08, 0.004)
        self.assertEqual(store.list_orders()[0]["status"], "partially_filled")
        pipeline.record_fill(result.receipt.order_id, 1.081, 0.006)
        self.assertEqual(store.list_orders()[0]["status"], "filled")
