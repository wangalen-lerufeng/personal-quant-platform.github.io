import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_platform.forward_validation import ForwardValidationTracker


class FakeProvider:
    def __init__(self, prices, index):
        self.prices = prices
        self.index = index

    def fetch_records(self, dataset, symbol, start_date, end_date):
        if dataset == "index_daily":
            return [{"trade_date": end_date.replace("-", ""), "close": self.index[symbol]}]
        return [{"trade_date": end_date.replace("-", ""), "close": self.prices[symbol]}]


class FakeDate:
    def __init__(self, today="2026-10-15"):
        self.today_value = today

    def today(self):
        from datetime import date as real_date

        return real_date.fromisoformat(self.today_value)

    @staticmethod
    def fromisoformat(value):
        from datetime import date as real_date

        return real_date.fromisoformat(value)


class ForwardValidationTests(unittest.TestCase):
    def test_start_equal_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            prices = {"600519.SH": 1500.0, "000858.SZ": 200.0}
            index = {"000300.SH": 4000.0}
            tracker = ForwardValidationTracker(store_path=str(Path(tmp) / "fv.json"))
            with patch("quant_platform.forward_validation.TushareDatasetProvider", return_value=FakeProvider(prices, index)):
                result = tracker.start(["600519.SH", "000858.SZ"])
            self.assertTrue(result["started"])
            self.assertEqual(result["weights"], {"600519.SH": 0.5, "000858.SZ": 0.5})
            self.assertEqual(result["strategy_nav"][0], 1.0)
            self.assertEqual(result["benchmark_nav"][0], 1.0)
            self.assertEqual(result["metrics"]["sample_days"], 1)
            self.assertFalse(result["execution_enabled"])

    def test_start_normalizes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            prices = {"600519.SH": 100.0, "000858.SZ": 100.0}
            tracker = ForwardValidationTracker(store_path=str(Path(tmp) / "fv.json"))
            with patch("quant_platform.forward_validation.TushareDatasetProvider", return_value=FakeProvider(prices, {"000300.SH": 1000.0})):
                result = tracker.start(["600519.SH", "000858.SZ"], weights=[1.0, 3.0])
            self.assertEqual(result["weights"], {"600519.SH": 0.25, "000858.SZ": 0.75})

    def test_update_marks_to_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            prices = {"600519.SH": 100.0, "000858.SZ": 100.0}
            index = {"000300.SH": 1000.0}
            fake_date = FakeDate()
            tracker = ForwardValidationTracker(store_path=str(Path(tmp) / "fv.json"))
            with patch("quant_platform.forward_validation.TushareDatasetProvider", return_value=FakeProvider(prices, index)), patch("quant_platform.forward_validation.date", fake_date):
                tracker.start(["600519.SH", "000858.SZ"])
                fake_date.today_value = "2026-10-16"
                prices["600519.SH"] = 110.0
                index["000300.SH"] = 1050.0
                result = tracker.update()
            self.assertAlmostEqual(result["metrics"]["strategy_cumulative_return"], 0.05, places=5)
            self.assertAlmostEqual(result["metrics"]["benchmark_cumulative_return"], 0.05, places=5)
            self.assertAlmostEqual(result["metrics"]["excess_return"], 0.0, places=5)
            self.assertEqual(result["metrics"]["sample_days"], 2)
            self.assertEqual(result["dates"], ["2026-10-15", "2026-10-16"])

    def test_start_requires_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ForwardValidationTracker(store_path=str(Path(tmp) / "fv.json"))
            with self.assertRaisesRegex(ValueError, "至少需要"):
                tracker.start([])

    def test_report_without_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ForwardValidationTracker(store_path=str(Path(tmp) / "fv.json"))
            result = tracker.report()
            self.assertFalse(result["started"])


if __name__ == "__main__":
    unittest.main()
