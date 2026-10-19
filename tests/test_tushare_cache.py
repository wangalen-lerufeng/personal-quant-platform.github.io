import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_platform.providers import (
    TushareDatasetProvider,
    _cache_load,
    _cache_store,
)


class TushareCacheTests(unittest.TestCase):
    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("quant_platform.providers._TUSHARE_CACHE_DIR", Path(directory)):
                records = [{"trade_date": "20240101", "close": 10.0}]
                _cache_store("daily", "600519.SH", "2024-01-01", "2024-01-10", records)
                self.assertEqual(
                    _cache_load("daily", "600519.SH", "2024-01-01", "2024-01-10"), records
                )
                # 不同日期范围不命中
                self.assertIsNone(
                    _cache_load("daily", "600519.SH", "2024-01-01", "2024-01-11")
                )

    def test_fetch_records_returns_cache_without_network(self):
        provider = TushareDatasetProvider("TEST_TOKEN")
        with patch("quant_platform.providers._cache_load", return_value=[{"x": 1}]), \
             patch("quant_platform.providers._create_tushare_client") as mock_client:
            result = provider.fetch_records("daily", "600519.SH", "2024-01-01", "2024-01-10")
        self.assertEqual(result, [{"x": 1}])
        mock_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
