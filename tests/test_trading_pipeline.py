import unittest

from quant_platform.adapters import Mt5PaperGateway
from quant_platform.adapters import QmtAdapter
from quant_platform.domain import AssetClass
from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.trading import TradingPipeline


class TradingPipelineTests(unittest.TestCase):
    def test_pipeline_executes_only_after_risk_approval(self):
        pipeline = TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway())
        intent = OrderIntent(symbol="EURUSD", side=OrderSide.BUY, quantity=0.01, strategy_id="demo")

        result = pipeline.submit(intent)

        self.assertTrue(result.allowed)
        self.assertIsNotNone(result.receipt)
        self.assertEqual(result.receipt.gateway, "mt5-paper")

    def test_pipeline_does_not_call_gateway_when_risk_rejects(self):
        gateway = Mt5PaperGateway()
        pipeline = TradingPipeline(RiskEngine(RiskLimits(kill_switch=True)), gateway)
        intent = OrderIntent(symbol="EURUSD", side=OrderSide.BUY, quantity=0.01, strategy_id="demo")

        result = pipeline.submit(intent)

        self.assertFalse(result.allowed)
        self.assertIsNone(result.receipt)

    def test_pipeline_propagates_adapter_rejection(self):
        pipeline = TradingPipeline(RiskEngine(RiskLimits()), QmtAdapter())
        intent = OrderIntent(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1,
            strategy_id="demo",
            asset_class=AssetClass.FX,
        )

        result = pipeline.submit(intent)

        self.assertFalse(result.allowed)
        self.assertEqual(result.decision.reason, "execution_rejected")
        self.assertEqual(result.receipt.status, "rejected")
