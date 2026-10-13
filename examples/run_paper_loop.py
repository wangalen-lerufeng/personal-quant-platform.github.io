"""Run a deterministic paper-trading loop without external broker dependencies."""

from datetime import datetime, timedelta, timezone

from quant_platform.adapters import Mt5PaperGateway
from quant_platform.domain import AssetClass, MarketEvent
from quant_platform.market_data import InMemoryMarketDataFeed
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.runtime import StrategyRuntime
from quant_platform.storage import SqliteAuditStore
from quant_platform.strategy import MomentumStrategy
from quant_platform.trading import TradingPipeline


def main() -> None:
    feed = InMemoryMarketDataFeed()
    store = SqliteAuditStore(":memory:")
    runtime = StrategyRuntime(
        feed,
        MomentumStrategy(lookback=2, quantity=0.01),
        TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway(), audit_store=store),
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for offset, price in enumerate((1.0800, 1.0810, 1.0830, 1.0850)):
        feed.publish(MarketEvent("EURUSD", AssetClass.FX, price, start + timedelta(days=offset)))

    print("paper results:", len(runtime.results))
    print("audit orders:", store.list_orders())


if __name__ == "__main__":
    main()
