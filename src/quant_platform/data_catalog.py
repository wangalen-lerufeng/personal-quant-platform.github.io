"""Versioned, auditable catalog for normalized market datasets."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .domain import AssetClass, MarketEvent


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    symbol_count: int
    start_time: str | None
    end_time: str | None
    duplicate_timestamps: int
    invalid_price_count: int
    missing_timestamp_count: int

    @property
    def has_warnings(self) -> bool:
        return any((self.duplicate_timestamps, self.invalid_price_count, self.missing_timestamp_count))

    @classmethod
    def from_events(cls, events: Iterable[MarketEvent]) -> "DataQualityReport":
        rows = list(events)
        timestamps = [(event.symbol, event.timestamp) for event in rows]
        duplicates = len(timestamps) - len(set(timestamps))
        invalid_prices = sum(1 for event in rows if event.price <= 0)
        missing_timestamps = sum(1 for event in rows if event.timestamp is None)
        ordered = sorted((event.timestamp for event in rows if event.timestamp is not None))
        return cls(
            row_count=len(rows),
            symbol_count=len({event.symbol for event in rows}),
            start_time=ordered[0].isoformat() if ordered else None,
            end_time=ordered[-1].isoformat() if ordered else None,
            duplicate_timestamps=max(duplicates, 0),
            invalid_price_count=invalid_prices,
            missing_timestamp_count=missing_timestamps,
        )


class DataCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"datasets": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def register(
        self,
        dataset_name: str,
        source: str,
        symbol: str,
        asset_class: AssetClass,
        events: Iterable[MarketEvent],
    ) -> dict:
        rows = list(events)
        canonical = [
            {"symbol": event.symbol, "asset_class": event.asset_class.value, "price": event.price, "timestamp": event.timestamp.isoformat()}
            for event in sorted(rows, key=lambda item: (item.symbol, item.timestamp))
        ]
        content_hash = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        catalog = self._read()
        versions = catalog.setdefault("datasets", {}).setdefault(dataset_name, [])
        version = {
            "version": len(versions) + 1,
            "source": source,
            "symbol": symbol,
            "asset_class": asset_class.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash,
            "quality": asdict(DataQualityReport.from_events(rows)),
        }
        versions.append(version)
        self.path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        return version

    def list_versions(self, dataset_name: str) -> list[dict]:
        return self._read().get("datasets", {}).get(dataset_name, [])
