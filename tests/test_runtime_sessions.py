from datetime import datetime, timezone
import unittest

from quant_platform.domain import AssetClass, MarketEvent
from quant_platform.market_data import InMemoryMarketDataFeed
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.runtime import StrategyRuntime
from quant_platform.strategy import MomentumStrategy
from quant_platform.trading import TradingPipeline
from quant_platform.adapters import QmtPaperGateway


class RuntimeSessionTests(unittest.TestCase):
    def test_runtime_rejects_unknown_market(self):
        feed = InMemoryMarketDataFeed()
        with self.assertRaises(ValueError):
            StrategyRuntime(
                feed,
                MomentumStrategy(lookback=2, quantity=1),
                TradingPipeline(RiskEngine(RiskLimits()), QmtPaperGateway()),
                market="unknown-market",
            )

    def test_runtime_skips_closed_mainland_equity_events(self):
        feed = InMemoryMarketDataFeed()
        runtime = StrategyRuntime(
            feed,
            MomentumStrategy(lookback=2, quantity=100),
            TradingPipeline(RiskEngine(RiskLimits()), QmtPaperGateway()),
            market="mainland_equity",
        )
        timestamps = [
            datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 4, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 4, 2, tzinfo=timezone.utc),
        ]
        for timestamp, price in zip(timestamps, (3.5, 3.51, 3.55)):
            feed.publish(MarketEvent("510300", AssetClass.ETF, price, timestamp))

        self.assertEqual(runtime.results, [])

    def test_runtime_skips_events_for_another_asset_class(self):
        feed = InMemoryMarketDataFeed()
        runtime = StrategyRuntime(
            feed,
            MomentumStrategy(lookback=2, quantity=0.01),
            TradingPipeline(RiskEngine(RiskLimits()), QmtPaperGateway()),
            market="mainland_equity",
        )
        base = datetime(2026, 1, 5, 2, 0, tzinfo=timezone.utc)
        for offset, price in enumerate((1.08, 1.081, 1.083)):
            feed.publish(MarketEvent("EURUSD", AssetClass.FX, price, base.replace(minute=offset)))

        self.assertEqual(runtime.results, [])
