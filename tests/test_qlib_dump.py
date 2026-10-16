import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant_platform.providers import TushareDatasetProvider
from quant_platform.qlib_dump import FIELDS, build_ohlcv_panel, dump_etf_to_qlib


def _synthetic_frame(symbols, n_days=40):
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rows = []
    for index, symbol in enumerate(symbols):
        for day_index, date in enumerate(dates):
            close = 4.0 * (index + 1) * (1 + 0.002 * day_index)
            rows.append({
                "symbol": symbol,
                "date": date.strftime("%Y%m%d"),
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "vwap": close,
                "volume": 1000.0,
                "amount": 1e6,
            })
    return pd.DataFrame(rows)


class BuildOhlcvPanelTests(unittest.TestCase):
    def test_build_ohlcv_panel_maps_fields_and_vwap(self):
        fake = [{
            "trade_date": "20260105", "open": 10, "high": 11, "low": 9,
            "close": 10.5, "vol": 1000, "amount": 20000,
        }]
        with patch("quant_platform.qlib_dump.TushareDatasetProvider") as MockProvider:
            MockProvider.return_value.fetch_records.return_value = fake
            frame = build_ohlcv_panel(["510300.SH"], "2026-01-01", "2026-01-10")
        self.assertEqual(frame.iloc[0]["symbol"], "510300.SH")
        # amount(千元)/vol(手)*10 = 20000/1000*10 = 200 元/份
        self.assertAlmostEqual(frame.iloc[0]["vwap"], 200.0)

    def test_build_ohlcv_panel_hk_vwap_has_no_factor(self):
        fake = [{
            "trade_date": "20260105", "open": 10, "high": 11, "low": 9,
            "close": 10.5, "vol": 1000, "amount": 20000,
        }]
        with patch("quant_platform.qlib_dump.TushareDatasetProvider") as MockProvider:
            MockProvider.return_value.fetch_records.return_value = fake
            frame = build_ohlcv_panel(["00700.HK"], "2026-01-01", "2026-01-10", asset_class="hk")
        self.assertAlmostEqual(frame.iloc[0]["vwap"], 20.0)

    def test_build_ohlcv_panel_stock_uses_daily_dataset(self):
        fake = [{
            "trade_date": "20260105", "open": 10, "high": 11, "low": 9,
            "close": 10.5, "vol": 1000, "amount": 20000,
        }]
        with patch("quant_platform.qlib_dump.TushareDatasetProvider") as MockProvider:
            MockProvider.return_value.fetch_records.return_value = fake
            build_ohlcv_panel(
                ["600519.SH", "000001.SZ"], "2026-01-01", "2026-01-10", asset_class="stock",
            )
            datasets = {call.args[0] for call in MockProvider.return_value.fetch_records.call_args_list}
        self.assertIn("daily", datasets)
        self.assertNotIn("fund_daily", datasets)

    def test_build_ohlcv_panel_hk_uses_hk_daily_dataset(self):
        fake = [{
            "trade_date": "20260105", "open": 10, "high": 11, "low": 9,
            "close": 10.5, "vol": 1000, "amount": 20000,
        }]
        with patch("quant_platform.qlib_dump.TushareDatasetProvider") as MockProvider:
            MockProvider.return_value.fetch_records.return_value = fake
            build_ohlcv_panel(
                ["00700.HK", "09988.HK"], "2026-01-01", "2026-01-10", asset_class="hk",
            )
            datasets = {call.args[0] for call in MockProvider.return_value.fetch_records.call_args_list}
        self.assertIn("hk_daily", datasets)
        self.assertNotIn("daily", datasets)


class FetchIndexConstituentsTests(unittest.TestCase):
    def test_returns_latest_date_codes(self):
        fake = [
            {"trade_date": "20260105", "con_code": "600519.SH"},
            {"trade_date": "20260105", "con_code": "000001.SZ"},
            {"trade_date": "20260106", "con_code": "600519.SH"},
            {"trade_date": "20260106", "con_code": "000002.SZ"},
        ]
        with patch("quant_platform.providers.TushareDatasetProvider.fetch_records", return_value=fake):
            provider = TushareDatasetProvider()
            codes = provider.fetch_index_constituents("000300.SH", "2026-01-10")
        self.assertEqual(codes, ["000002.SZ", "600519.SH"])


class DumpEtfToQlibTests(unittest.TestCase):
    def test_dump_produces_qlib_layout_and_is_readable(self):
        from qlib.data import D

        from quant_platform.qlib_dump import init_qlib

        frame = _synthetic_frame(["510300.SH", "510500.SH"], n_days=40)
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = dump_etf_to_qlib(frame, Path(tmp) / "qlib_data")
            self.assertTrue((Path(qlib_dir) / "calendars" / "day.txt").exists())
            self.assertTrue((Path(qlib_dir) / "instruments" / "all.txt").exists())
            for field in FIELDS:
                self.assertTrue(
                    (Path(qlib_dir) / "features" / "510300.sh" / f"{field}.day.bin").exists()
                )
            init_qlib(qlib_dir)
            data = D.features(
                ["510300.SH"], ["$close"], start_time="2026-01-05", end_time="2026-01-15",
            )
            self.assertFalse(data.empty)


if __name__ == "__main__":
    unittest.main()
