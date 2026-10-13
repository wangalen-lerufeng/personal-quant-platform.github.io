from .domain import OrderIntent, OrderReceipt
from .execution import OrderState


class AdapterRegistry:
    def __init__(self, adapters):
        self.adapters = list(adapters)
        markets = [adapter.market for adapter in self.adapters]
        if len(markets) != len(set(markets)):
            raise ValueError("duplicate adapter market")

    def for_market(self, market: str):
        for adapter in self.adapters:
            if adapter.market == market:
                return adapter
        raise KeyError(f"unknown adapter market: {market}")

    def publish_market_data(self, market: str, feed, events) -> int:
        return self.for_market(market).publish_market_data(feed, events)

    def submit_order(self, market: str, intent: OrderIntent) -> OrderReceipt:
        return self.for_market(market).submit_order(intent)

    def cancel_order(self, market: str, order_id: str) -> OrderState:
        return self.for_market(market).cancel_order(order_id)

    def get_positions(self, market: str) -> dict[str, float]:
        return self.for_market(market).get_positions()

    def health(self) -> list[dict[str, object]]:
        return [adapter.health_check() for adapter in self.adapters]
