import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_platform.providers import (
    TushareDatasetProvider,
    _cache_load_full,
    _cache_store_full,
)


class TushareCacheTests(unittest.TestCase):
    def test_cache_full_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("quant_platform.providers._TUSHARE_CACHE_DIR", Path(directory)):
                records = [{"trade_date": "20240101", "close": 10.0}]
                _cache_store_full("daily", "600519.SH", records)
                self.assertEqual(_cache_load_full("daily", "600519.SH"), records)

    def test_fetch_records_returns_cache_without_network(self):
        cached = [
            {"trade_date": "20240101", "close": 10.0},
            {"trade_date": "20240110", "close": 11.0},
        ]
        provider = TushareDatasetProvider("TEST_TOKEN")
        with patch("quant_platform.providers._cache_load_full", return_value=cached), \
             patch("quant_platform.providers._create_tushare_client") as mock_client:
            result = provider.fetch_records("daily", "600519.SH", "2024-01-01", "2024-01-10")
        self.assertEqual([r["trade_date"] for r in result], ["20240101", "20240110"])
        mock_client.assert_not_called()

    def test_fetch_records_only_pulls_tail_incrementally(self):
        cached = [
            {"trade_date": "20240101", "close": 10.0},
            {"trade_date": "20240102", "close": 11.0},
        ]
        provider = TushareDatasetProvider("TEST_TOKEN")
        with patch("quant_platform.providers._cache_load_full", return_value=cached), \
             patch.object(provider, "_fetch_tushare", return_value=[{"trade_date": "20240103", "close": 12.0}]) as mock_fetch, \
             patch("quant_platform.providers._cache_store_full") as mock_store:
            result = provider.fetch_records("daily", "600519.SH", "2024-01-01", "2024-01-03")

        # 增量拉取起点应为缓存最新日期 20240102 的次日 20240103
        self.assertEqual(mock_fetch.call_args[0][2], "20240103")
        self.assertEqual(mock_fetch.call_args[0][3], "20240103")
        self.assertEqual(
            [r["trade_date"] for r in result], ["20240101", "20240102", "20240103"]
        )
        mock_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
