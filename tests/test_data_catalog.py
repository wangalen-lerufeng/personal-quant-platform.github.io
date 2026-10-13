import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_platform.data_catalog import DataCatalog, DataQualityReport
from quant_platform.domain import AssetClass, MarketEvent


class DataCatalogTests(unittest.TestCase):
    def test_quality_report_detects_duplicates_and_gaps(self):
        events = [
            MarketEvent("510300.SH", AssetClass.ETF, 3.1, datetime(2025, 1, 1, tzinfo=timezone.utc)),
            MarketEvent("510300.SH", AssetClass.ETF, 3.1, datetime(2025, 1, 1, tzinfo=timezone.utc)),
            MarketEvent("510300.SH", AssetClass.ETF, 3.2, datetime(2025, 1, 3, tzinfo=timezone.utc)),
        ]
        report = DataQualityReport.from_events(events)
        self.assertEqual(report.row_count, 3)
        self.assertEqual(report.duplicate_timestamps, 1)
        self.assertEqual(report.invalid_price_count, 0)
        self.assertTrue(report.has_warnings)

    def test_catalog_registers_version_with_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = DataCatalog(Path(directory) / "catalog.json")
            version = catalog.register(
                dataset_name="etf_daily",
                source="tushare",
                symbol="510300.SH",
                asset_class=AssetClass.ETF,
                events=[MarketEvent("510300.SH", AssetClass.ETF, 3.1, datetime(2025, 1, 1, tzinfo=timezone.utc))],
            )
            self.assertEqual(version["version"], 1)
            self.assertEqual(len(version["content_hash"]), 64)
            self.assertEqual(len(catalog.list_versions("etf_daily")), 1)

