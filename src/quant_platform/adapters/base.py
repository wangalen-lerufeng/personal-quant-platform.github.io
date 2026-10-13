from collections.abc import Iterable

from ..domain import AssetClass, OrderIntent, OrderReceipt
from ..execution import OrderState
from ..market_data import InMemoryMarketDataFeed
from .paper import _PaperGateway


class MarketAdapter(_PaperGateway):
    market = "unknown"
    mode = "paper"
    live_trading_enabled = False
    supported_asset_class: AssetClass | None = None

    def submit_order(self, intent: OrderIntent) -> OrderReceipt:
        if self.supported_asset_class is not None and intent.asset_class not in (None, self.supported_asset_class):
            return OrderReceipt(symbol=intent.symbol, status="rejected", gateway=self.name)
        return super().submit_order(intent)

    def cancel_order(self, order_id: str) -> OrderState:
        """Paper cancellation hook; real adapters will implement broker semantics."""
        return super().cancel_order(order_id)

    def get_positions(self) -> dict[str, float]:
        return {}

    def publish_market_data(self, feed: InMemoryMarketDataFeed, events: Iterable) -> int:
        """Paper-only market-data hook shared by future SDK integrations."""
        events = list(events)
        for event in events:
            if event.asset_class is not self.supported_asset_class:
                raise ValueError(f"{self.name} cannot publish {event.asset_class.value} events")
        for event in events:
            feed.publish(event)
        return len(events)

    def health_check(self) -> dict[str, object]:
        return {
            "market": self.market,
            "gateway": self.name,
            "supported_asset_class": self.supported_asset_class.value if self.supported_asset_class else None,
            "mode": self.mode,
            "live_trading_enabled": self.live_trading_enabled,
            "execution_enabled": False,
            "status": "ready",
        }
