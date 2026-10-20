import unittest

import pandas as pd

from quant_platform.research_universe import (
    build_dynamic_universe_index,
    rank_scores_within_industry,
)


class DynamicUniverseTests(unittest.TestCase):
    def _ohlcv(self):
        rows = []
        dates = pd.bdate_range("2024-01-01", periods=45)
        for day_no, day in enumerate(dates):
            for symbol, amount in (("A", 300.0), ("B", 200.0), ("C", 100.0)):
                rows.append({"symbol": symbol, "date": day, "amount": amount + day_no})
        return pd.DataFrame(rows)

    def test_dynamic_universe_uses_trailing_liquidity_and_month_end(self):
        index = build_dynamic_universe_index(
            self._ohlcv(), liquidity_window=5, top_n=2, min_history=5, monthly=True,
        )
        dates = index.get_level_values("datetime")
        self.assertTrue(all(day == day.to_period("M").end_time.normalize() or day.weekday() < 5 for day in dates))
        self.assertLessEqual(pd.Series(index.get_level_values("instrument"), index=dates).groupby(level=0).nunique().max(), 2)
        self.assertNotIn("C", set(index.get_level_values("instrument")))

    def test_future_amount_changes_do_not_change_past_membership(self):
        frame = self._ohlcv()
        original = build_dynamic_universe_index(
            frame, liquidity_window=5, top_n=2, min_history=5, monthly=False,
        )
        cutoff = pd.Timestamp("2024-01-31")
        changed = frame.copy()
        changed.loc[changed["date"] > cutoff, "amount"] = 10_000_000
        revised = build_dynamic_universe_index(
            changed, liquidity_window=5, top_n=2, min_history=5, monthly=False,
        )
        self.assertEqual(
            set(original[original.get_level_values("datetime") <= cutoff]),
            set(revised[revised.get_level_values("datetime") <= cutoff]),
        )

    def test_rank_scores_within_industry(self):
        scores = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-31"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "score": [1.0, 2.0, 10.0, 20.0],
        })
        ranked = rank_scores_within_industry(
            scores, {"A": "bank", "B": "bank", "C": "tech", "D": "tech"}
        )
        self.assertEqual(ranked["score"].tolist(), [0.5, 1.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
