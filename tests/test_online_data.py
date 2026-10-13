import json
import os
import unittest
from datetime import timezone

from quant_platform.domain import AssetClass
from quant_platform.online_data import RestMarketDataClient


class Response:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def read(self): return json.dumps(self.payload).encode()


class OnlineDataTests(unittest.TestCase):
    def test_fetch_maps_provider_rows_and_auth_header(self):
        captured = {}
        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            return Response({"data": [{"symbol": "EURUSD", "price": 1.08, "timestamp": "2026-01-01T00:00:00+00:00"}]})
        os.environ["TEST_MARKET_KEY"] = "secret"
        try:
            events = RestMarketDataClient("https://quotes.test/v1", "TEST_MARKET_KEY", opener=opener).fetch(["EURUSD"], AssetClass.FX)
        finally:
            os.environ.pop("TEST_MARKET_KEY", None)
        self.assertEqual(events[0].symbol, "EURUSD")
        self.assertEqual(events[0].timestamp.tzinfo, timezone.utc)
        self.assertEqual(captured["auth"], "Bearer secret")
        self.assertIn("symbols=EURUSD", captured["url"])

    def test_client_requires_url_and_rejects_bad_payload(self):
        with self.assertRaises(ValueError):
            RestMarketDataClient("")

    def test_transient_failures_use_bounded_retries_and_report_recovery(self):
        attempts = []
        health = []

        def opener(_request, timeout):
            attempts.append(timeout)
            if len(attempts) < 3:
                raise TimeoutError("temporary outage")
            return Response({"data": [{"symbol": "EURUSD", "price": 1.08, "timestamp": "2026-01-01T00:00:00+00:00"}]})

        client = RestMarketDataClient(
            "https://quotes.test/v1",
            timeout=1,
            opener=opener,
            max_attempts=3,
            retry_delay_seconds=0,
            health_callback=health.append,
        )

        events = client.fetch(["EURUSD"], AssetClass.FX)

        self.assertEqual(len(events), 1)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(health, [False, False, True])

    def test_retry_budget_is_finite_when_provider_stays_down(self):
        attempts = []

        def opener(_request, timeout):
            self.assertEqual(timeout, 10.0)
            attempts.append(1)
            raise TimeoutError("provider down")

        client = RestMarketDataClient(
            "https://quotes.test/v1",
            opener=opener,
            max_attempts=2,
            retry_delay_seconds=0,
        )

        with self.assertRaisesRegex(TimeoutError, "provider down"):
            client.fetch(["EURUSD"], AssetClass.FX)

        self.assertEqual(len(attempts), 2)
