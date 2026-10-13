from datetime import datetime, timedelta, timezone
import unittest

from quant_platform.adapters import Mt5PaperGateway
from quant_platform.domain import AssetClass, MarketEvent
from quant_platform.market_data import InMemoryMarketDataFeed
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.runtime import StrategyRuntime
from quant_platform.strategy import MomentumStrategy
from quant_platform.trading import TradingPipeline


class CapturingGateway(Mt5PaperGateway):
    def submit_order(self, intent):
        self.last_intent = intent
        return super().submit_order(intent)


class CapturingRiskEngine(RiskEngine):
    def evaluate(self, intent):
        self.last_intent = intent
        return super().evaluate(intent)


class RuntimeTests(unittest.TestCase):
    def test_runtime_turns_breakout_event_into_paper_order(self):
        feed = InMemoryMarketDataFeed()
        gateway = Mt5PaperGateway()
        runtime = StrategyRuntime(
            feed,
            MomentumStrategy(lookback=2, quantity=0.01),
            TradingPipeline(RiskEngine(RiskLimits()), gateway),
        )
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for offset, price in enumerate((1.08, 1.081, 1.083)):
            feed.publish(MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=offset)))

        self.assertEqual(len(runtime.results), 1)
        self.assertTrue(runtime.results[0].allowed)
        self.assertEqual(runtime.results[0].receipt.gateway, "mt5-paper")

    def test_runtime_passes_event_context_into_order_intent(self):
        feed = InMemoryMarketDataFeed()
        gateway = CapturingGateway()
        runtime = StrategyRuntime(
            feed,
            MomentumStrategy(lookback=2, quantity=0.01),
            TradingPipeline(CapturingRiskEngine(RiskLimits(max_order_value=0.01)), gateway),
        )
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for offset, price in enumerate((1.08, 1.081, 1.083)):
            feed.publish(MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=offset)))

        self.assertFalse(runtime.results[0].allowed)
        self.assertEqual(runtime.results[0].decision.reason, "order_value_exceeded")
        self.assertEqual(runtime.pipeline.risk_engine.last_intent.asset_class, AssetClass.FX)
        self.assertEqual(runtime.pipeline.risk_engine.last_intent.reference_price, 1.083)

    def test_runtime_can_replay_preloaded_feed_events(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        feed = InMemoryMarketDataFeed(
            [
                MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
                for i, price in enumerate((1.08, 1.081, 1.083))
            ]
        )
        runtime = StrategyRuntime(
            feed,
            MomentumStrategy(lookback=2, quantity=0.01),
            TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway()),
            replay=True,
        )

        self.assertEqual(len(runtime.results), 1)

    def test_runtime_close_unsubscribes_from_feed(self):
        feed = InMemoryMarketDataFeed()
        runtime = StrategyRuntime(
            feed,
            MomentumStrategy(lookback=2, quantity=0.01),
            TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway()),
        )
        runtime.close()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for offset, price in enumerate((1.08, 1.081, 1.083)):
            feed.publish(MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=offset)))

        self.assertEqual(runtime.results, [])
