from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quant_platform.domain import AssetClass
from quant_platform.market_data import CsvMarketDataLoader


class CsvDataTests(unittest.TestCase):
    def test_loader_reads_standard_market_event_csv(self):
        content = "timestamp,symbol,asset_class,price\n2026-01-01T00:00:00+00:00,EURUSD,fx,1.08\n"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "eurusd.csv"
            path.write_text(content)

            events = CsvMarketDataLoader().load(path)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].asset_class, AssetClass.FX)
        self.assertEqual(events[0].price, 1.08)
