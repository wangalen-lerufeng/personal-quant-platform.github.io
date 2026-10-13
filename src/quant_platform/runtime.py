from .domain import MarketEvent, OrderIntent
from .market_data import InMemoryMarketDataFeed
from .strategy import MomentumStrategy
from .trading import PipelineResult, TradingPipeline
from .market_calendar import is_trading_time
from .domain import AssetClass


_MARKET_ASSET_CLASSES = {
    "mainland_equity": AssetClass.ETF,
    "mainland_futures": AssetClass.FUTURES,
    "mt5_broker_symbols": AssetClass.FX,
}


class StrategyRuntime:
    def __init__(
        self,
        feed: InMemoryMarketDataFeed,
        strategy: MomentumStrategy,
        pipeline: TradingPipeline,
        market: str | None = None,
        replay: bool = False,
    ):
        self.strategy = strategy
        self.pipeline = pipeline
        self.results: list[PipelineResult] = []
        self.market = market
        self._feed = feed
        self._closed = False
        if market is not None and market not in _MARKET_ASSET_CLASSES:
            raise ValueError(f"unknown market: {market}")
        feed.subscribe(self.on_event, replay=replay)

    def close(self) -> None:
        if not self._closed:
            self._feed.unsubscribe(self.on_event)
            self._closed = True

    def on_event(self, event: MarketEvent) -> None:
        expected_asset_class = _MARKET_ASSET_CLASSES.get(self.market) if self.market is not None else None
        if expected_asset_class is not None and expected_asset_class is not event.asset_class:
            return
        if self.market is not None and not is_trading_time(self.market, event.timestamp):
            return
        signal = self.strategy.on_event(event)
        if signal is None:
            return
        intent = OrderIntent.from_signal(
            signal,
            strategy_id=self.strategy.__class__.__name__,
            asset_class=event.asset_class,
            reference_price=event.price,
        )
        self.results.append(self.pipeline.submit(intent))
