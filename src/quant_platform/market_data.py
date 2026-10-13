from collections.abc import Callable, Iterable
import csv
from datetime import datetime
from pathlib import Path

from .domain import AssetClass, MarketEvent
from .data_quality import validate_events


class InMemoryMarketDataFeed:
    def __init__(self, events: Iterable[MarketEvent] | None = None):
        self.events = list(events or [])
        validate_events(self.events)
        self._subscribers: list[Callable[[MarketEvent], None]] = []

    def subscribe(self, callback: Callable[[MarketEvent], None], replay: bool = False) -> None:
        already_subscribed = callback in self._subscribers
        if not already_subscribed:
            self._subscribers.append(callback)
        if replay and not already_subscribed:
            for event in self.events:
                callback(event)

    def unsubscribe(self, callback: Callable[[MarketEvent], None]) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            return

    def publish(self, event: MarketEvent) -> None:
        validate_events(self.events + [event])
        self.events.append(event)
        for callback in list(self._subscribers):
            callback(event)

    def latest_prices(self) -> dict[str, float]:
        """Return the latest validated price observed for each symbol."""
        latest: dict[str, float] = {}
        for event in self.events:
            latest[event.symbol] = event.price
        return latest


class CsvMarketDataLoader:
    def load(self, path: str | Path) -> list[MarketEvent]:
        events = []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                events.append(
                    MarketEvent(
                        symbol=row["symbol"],
                        asset_class=AssetClass(row["asset_class"]),
                        price=float(row["price"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                )
        validate_events(events)
        return events
