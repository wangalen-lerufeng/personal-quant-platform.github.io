import unittest

import pandas as pd

from quant_platform.etf_universe import (
    add_forward_returns,
    apply_adjustment_factor,
    build_etf_panel,
    drop_suspended,
    liquid_symbols,
    tradable_pool,
)
from quant_platform.factor_analysis import analyze_factor_panel


class ApplyAdjustmentFactorTests(unittest.TestCase):
    def test_adjusted_close_is_close_times_factor(self):
        daily = pd.DataFrame({
            "symbol": ["A", "A", "B"],
            "trade_date": ["20260101", "20260102", "20260101"],
            "close": [10.0, 11.0, 20.0],
            "volume": [100, 100, 200],
        })
        adj_factors = pd.DataFrame({
            "symbol": ["A", "A"],
            "trade_date": ["20260101", "20260102"],
            "adj_factor": [1.0, 0.5],
        })
        result = apply_adjustment_factor(daily, adj_factors)
        self.assertEqual(list(result["symbol"]), ["A", "A"])
        self.assertEqual(list(result["adj_close"]), [10.0, 5.5])

    def test_rows_without_adjustment_factor_are_dropped(self):
        daily = pd.DataFrame({
            "symbol": ["A", "B"],
            "trade_date": ["20260101", "20260101"],
            "close": [10.0, 20.0],
        })
        adj_factors = pd.DataFrame({
            "symbol": ["A"],
            "trade_date": ["20260101"],
            "adj_factor": [1.0],
        })
        result = apply_adjustment_factor(daily, adj_factors)
        self.assertEqual(list(result["symbol"]), ["A"])

    def test_non_price_columns_are_preserved(self):
        daily = pd.DataFrame({
            "symbol": ["A"],
            "trade_date": ["20260101"],
            "close": [10.0],
            "volume": [100],
            "amount": [1000.0],
        })
        adj_factors = pd.DataFrame({
            "symbol": ["A"],
            "trade_date": ["20260101"],
            "adj_factor": [2.0],
        })
        result = apply_adjustment_factor(daily, adj_factors)
        self.assertEqual(result.loc[0, "volume"], 100)
        self.assertEqual(result.loc[0, "amount"], 1000.0)


class TradablePoolTests(unittest.TestCase):
    def test_keeps_only_listed_status(self):
        basics = [
            {"ts_code": "510300.SH", "list_status": "L"},
            {"ts_code": "510500.SH", "list_status": "D"},
            {"ts_code": "159919.SZ", "list_status": "P"},
        ]
        self.assertEqual(tradable_pool(basics), ["510300.SH"])

    def test_deduplicates_and_sorts(self):
        basics = [
            {"ts_code": "159919.SZ", "list_status": "L"},
            {"ts_code": "510300.SH", "list_status": "L"},
            {"ts_code": "510300.SH", "list_status": "L"},
        ]
        self.assertEqual(tradable_pool(basics), ["159919.SZ", "510300.SH"])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(tradable_pool([]), [])

    def test_rows_without_list_status_are_excluded(self):
        basics = [
            {"ts_code": "510300.SH", "list_status": "L"},
            {"ts_code": "510500.SH"},
        ]
        self.assertEqual(tradable_pool(basics), ["510300.SH"])


class DropSuspendedTests(unittest.TestCase):
    def test_drops_zero_and_missing_volume(self):
        daily = pd.DataFrame({
            "symbol": ["A", "A", "B", "C"],
            "trade_date": ["20260101", "20260102", "20260101", "20260101"],
            "volume": [100, 0, 200, None],
        })
        result = drop_suspended(daily)
        self.assertEqual(list(result["symbol"]), ["A", "B"])

    def test_requires_volume_column(self):
        daily = pd.DataFrame({"symbol": ["A"], "trade_date": ["20260101"]})
        with self.assertRaisesRegex(ValueError, "volume"):
            drop_suspended(daily)


class LiquidSymbolsTests(unittest.TestCase):
    def test_keeps_symbols_above_median_threshold(self):
        daily = pd.DataFrame({
            "symbol": ["A"] * 3 + ["B"] * 3,
            "trade_date": ["20260101", "20260102", "20260103"] * 2,
            "amount": [100, 100, 100, 10, 10, 10],
        })
        self.assertEqual(liquid_symbols(daily, window=3, min_median_amount=50), ["A"])

    def test_window_limits_trailing_history(self):
        daily = pd.DataFrame({
            "symbol": ["A", "A", "A"],
            "trade_date": ["20260101", "20260102", "20260103"],
            "amount": [0, 0, 1000],
        })
        self.assertEqual(liquid_symbols(daily, window=1, min_median_amount=50), ["A"])
        self.assertEqual(liquid_symbols(daily, window=3, min_median_amount=50), [])


class AddForwardReturnsTests(unittest.TestCase):
    def test_forward_return_1d_values_and_tail_nan(self):
        daily = pd.DataFrame({
            "symbol": ["A", "A", "A"],
            "trade_date": ["20260101", "20260102", "20260103"],
            "adj_close": [100.0, 110.0, 121.0],
        })
        result = add_forward_returns(daily, horizons=(1,))
        self.assertAlmostEqual(result.loc[0, "forward_return_1d"], 0.10)
        self.assertAlmostEqual(result.loc[1, "forward_return_1d"], 0.10)
        self.assertTrue(pd.isna(result.loc[2, "forward_return_1d"]))

    def test_column_names_contain_forward(self):
        daily = pd.DataFrame({
            "symbol": ["A", "A"],
            "trade_date": ["20260101", "20260102"],
            "adj_close": [100.0, 110.0],
        })
        result = add_forward_returns(daily, horizons=(1, 5, 20))
        for column in ("forward_return_1d", "forward_return_5d", "forward_return_20d"):
            self.assertIn("forward", column)
            self.assertIn(column, result.columns)

    def test_shifts_are_per_symbol(self):
        daily = pd.DataFrame({
            "symbol": ["A", "A", "B", "B"],
            "trade_date": ["20260101", "20260102", "20260101", "20260102"],
            "adj_close": [100.0, 110.0, 50.0, 55.0],
        })
        result = add_forward_returns(daily, horizons=(1,))
        self.assertAlmostEqual(result.loc[0, "forward_return_1d"], 0.10)
        self.assertAlmostEqual(result.loc[2, "forward_return_1d"], 0.10)
        self.assertTrue(pd.isna(result.loc[1, "forward_return_1d"]))
        self.assertTrue(pd.isna(result.loc[3, "forward_return_1d"]))


class BuildEtfPanelTests(unittest.TestCase):
    def _sample_inputs(self):
        symbols = ["510300.SH", "510500.SH", "159919.SZ"]
        bases = {"510300.SH": 4.0, "510500.SH": 7.0, "159919.SZ": 10.0}
        dates = ["20260105", "20260106", "20260107", "20260108", "20260109"]
        daily_rows = []
        adj_rows = []
        for symbol in symbols:
            base = bases[symbol]
            for index, date in enumerate(dates):
                close = base + index
                daily_rows.append({
                    "symbol": symbol,
                    "trade_date": date,
                    "close": close,
                    "volume": 1000,
                    "amount": 1_000_000,
                })
                adj_rows.append({"symbol": symbol, "trade_date": date, "adj_factor": 1.0})
        basics = [
            {"ts_code": "510300.SH", "list_status": "L"},
            {"ts_code": "510500.SH", "list_status": "L"},
            {"ts_code": "159919.SZ", "list_status": "D"},
        ]
        return (
            pd.DataFrame(daily_rows),
            pd.DataFrame(adj_rows),
            basics,
        )

    def test_panel_has_expected_columns_sorted_and_suspended_removed(self):
        daily, adj, basics = self._sample_inputs()
        panel = build_etf_panel(daily, adj, basics=basics, horizons=(1, 5))
        expected = {
            "timestamp", "symbol", "adj_close", "volume", "amount",
            "forward_return_1d", "forward_return_5d",
        }
        self.assertEqual(set(panel.columns), expected)
        self.assertNotIn("159919.SZ", set(panel["symbol"]))
        self.assertTrue((panel["timestamp"].is_monotonic_increasing))

    def test_panel_drops_suspended_rows(self):
        daily, adj, basics = self._sample_inputs()
        daily.loc[daily["trade_date"] == "20260106", "volume"] = 0
        panel = build_etf_panel(daily, adj, basics=basics, horizons=(1,))
        self.assertNotIn("20260106", set(panel["timestamp"].dt.strftime("%Y%m%d")))

    def test_panel_applies_liquidity_filter(self):
        daily, adj, basics = self._sample_inputs()
        daily.loc[daily["symbol"] == "510500.SH", "amount"] = 1.0
        panel = build_etf_panel(
            daily, adj, basics=basics, horizons=(1,), min_median_amount=100_000,
        )
        self.assertNotIn("510500.SH", set(panel["symbol"]))

    def test_panel_feeds_factor_analysis_contract(self):
        daily, adj, basics = self._sample_inputs()
        panel = build_etf_panel(daily, adj, basics=basics, horizons=(1, 5))
        panel["factor"] = panel["adj_close"]
        report = analyze_factor_panel(
            panel,
            factor_column="factor",
            return_columns=["forward_return_1d", "forward_return_5d"],
            date_column="timestamp",
            asset_column="symbol",
        )
        self.assertFalse(report["future_data_used_for_feature"])
        self.assertGreater(report["date_count"], 1)
        self.assertIn("forward_return_1d", report["horizons"])


if __name__ == "__main__":
    unittest.main()
