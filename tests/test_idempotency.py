import unittest

from quant_platform.adapters import Mt5PaperGateway
from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.trading import TradingPipeline
from quant_platform.storage import SqliteAuditStore


class IdempotencyTests(unittest.TestCase):
    def test_same_client_order_id_is_not_submitted_twice(self):
        pipeline = TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway())
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo", client_order_id="demo-1")

        first = pipeline.submit(intent)
        second = pipeline.submit(intent)

        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertEqual(second.decision.reason, "duplicate_order")

    def test_same_client_order_id_is_rejected_after_pipeline_recreation(self):
        store = SqliteAuditStore(":memory:")
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo", client_order_id="persistent-1")
        TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway(), audit_store=store).submit(intent)

        second = TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway(), audit_store=store).submit(intent)

        self.assertFalse(second.allowed)
        self.assertEqual(second.decision.reason, "duplicate_order")
