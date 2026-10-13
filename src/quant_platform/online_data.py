"""Generic, read-only REST market-data client.

The client deliberately has no order or broker methods. Providers only need to
return rows containing symbol, asset_class, price and timestamp (or configure
the field names in the constructor).
"""

import json
import os
import ssl
import time
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from .domain import AssetClass, MarketEvent
from .data_quality import validate_events


def _secure_urlopen(request, timeout):
    return urlopen(request, timeout=timeout, context=ssl.create_default_context(cafile=certifi.where()))


class RestMarketDataClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str = "MARKET_DATA_API_KEY",
        timeout: float = 10.0,
        opener=_secure_urlopen,
        max_attempts: int = 1,
        retry_delay_seconds: float = 0.0,
        health_callback=None,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self._opener = opener
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.health_callback = health_callback

    def fetch(self, symbols: list[str], asset_class: AssetClass, params: dict[str, str] | None = None) -> list[MarketEvent]:
        query = dict(params or {})
        query["symbols"] = ",".join(symbols)
        url = self.base_url + ("&" if "?" in self.base_url else "?") + urlencode(query)
        headers = {"Accept": "application/json"}
        api_key = os.getenv(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(url, headers=headers, method="GET")
        for attempt in range(self.max_attempts):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if self.health_callback is not None:
                    self.health_callback(True)
                break
            except OSError:
                if self.health_callback is not None:
                    self.health_callback(False)
                if attempt + 1 >= self.max_attempts:
                    raise
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        events = [MarketEvent(row["symbol"], asset_class, float(row["price"]), datetime.fromisoformat(row["timestamp"])) for row in rows]
        validate_events(events)
        return events
