from .domain import MarketEvent


class DataQualityError(ValueError):
    pass


def validate_events(events: list[MarketEvent]) -> int:
    previous_by_symbol = {}
    asset_class_by_symbol = {}
    seen: set[tuple[str, object]] = set()
    for event in events:
        key = (event.symbol, event.timestamp)
        if key in seen:
            raise DataQualityError(f"duplicate event: {event.symbol} at {event.timestamp}")
        previous = previous_by_symbol.get(event.symbol)
        previous_asset_class = asset_class_by_symbol.get(event.symbol)
        if previous_asset_class is not None and previous_asset_class is not event.asset_class:
            raise DataQualityError(f"asset class changed for symbol: {event.symbol}")
        if previous is not None and event.timestamp < previous:
            raise DataQualityError("events are out of order")
        if event.price <= 0:
            raise DataQualityError("event price must be positive")
        seen.add(key)
        previous_by_symbol[event.symbol] = event.timestamp
        asset_class_by_symbol[event.symbol] = event.asset_class
    return len(events)
