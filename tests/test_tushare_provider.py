import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from quant_platform.domain import AssetClass
from quant_platform.providers import TushareStockDailyProvider


class TushareCustomEndpointTests(unittest.TestCase):
    def test_stock_daily_uses_token_and_custom_sdk_endpoint_from_environment(self):
        client = Mock()
        client.daily.return_value = pd.DataFrame([
            {"trade_date": "20260105", "close": 10.5},
        ])
        tushare = SimpleNamespace(pro_api=Mock(return_value=client))
        environment = {
            "TUSHARE_TOKEN": "test-token-not-a-real-secret",
            "TUSHARE_BASE_URL": "http://data.example.test:8030/dataapi",
        }

        with patch.dict(os.environ, environment, clear=False), patch.dict(
            sys.modules, {"tushare": tushare}
        ):
            events = TushareStockDailyProvider().fetch(
                "000001.SZ", "2026-01-01", "2026-01-05"
            )

        tushare.pro_api.assert_called_once_with("test-token-not-a-real-secret")
        self.assertEqual(client._DataApi__http_url, environment["TUSHARE_BASE_URL"])
        self.assertEqual(events[0].asset_class, AssetClass.EQUITY)

    def test_custom_endpoint_rejects_non_http_urls(self):
        client = Mock()
        tushare = SimpleNamespace(pro_api=Mock(return_value=client))
        with patch.dict(
            os.environ,
            {"TUSHARE_TOKEN": "test-token", "TUSHARE_BASE_URL": "file:///tmp/api"},
            clear=False,
        ), patch.dict(sys.modules, {"tushare": tushare}):
            with self.assertRaisesRegex(RuntimeError, "http"):
                TushareStockDailyProvider().fetch(
                    "000001.SZ", "2026-01-01", "2026-01-05"
                )


if __name__ == "__main__":
    unittest.main()
