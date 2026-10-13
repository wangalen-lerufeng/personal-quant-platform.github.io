from datetime import datetime, timedelta, timezone

from .domain import MarketEvent, OrderSide, Signal


class MomentumStrategy:
    def __init__(self, lookback: int, quantity: float, exit_on_negative: bool = False):
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.lookback = lookback
        self.quantity = quantity
        self.exit_on_negative = exit_on_negative
        self._prices: dict[str, list[float]] = {}
        self._entered: set[str] = set()

    @staticmethod
    def timestamp(index: int) -> datetime:
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)

    def on_event(self, event: MarketEvent) -> Signal | None:
        prices = self._prices.setdefault(event.symbol, [])
        previous = prices[-self.lookback :]
        prices.append(event.price)
        if len(previous) < self.lookback:
            return None
        if event.symbol in self._entered:
            if self.exit_on_negative and event.price < min(previous):
                self._entered.discard(event.symbol)
                return Signal(event.symbol, OrderSide.SELL, self.quantity, "negative_momentum_exit")
            return None
        if event.price <= max(previous):
            return None
        self._entered.add(event.symbol)
        return Signal(event.symbol, OrderSide.BUY, self.quantity, "upward_breakout")

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._prices.clear()
            self._entered.clear()
            return
        self._entered.discard(symbol)
