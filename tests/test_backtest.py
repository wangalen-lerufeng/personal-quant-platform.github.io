from datetime import datetime, timedelta, timezone
import unittest

from quant_platform.backtest import (
    BacktestEngine,
    FuturesRolloverRule,
    MarketExecutionPolicy,
    chronological_split,
)
from quant_platform.domain import AssetClass, MarketEvent, OrderSide, Signal
from quant_platform.strategy import MomentumStrategy


class BacktestTests(unittest.TestCase):
    def test_backtest_returns_trade_count_and_positive_result(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
            for i, price in enumerate((1.08, 1.081, 1.083, 1.085))
        ]

        result = BacktestEngine(MomentumStrategy(lookback=2, quantity=1)).run(events)

        self.assertEqual(result.trade_count, 1)
        self.assertEqual(result.signal_count, 1)
        self.assertGreater(result.total_return, 0)
        self.assertEqual(result.win_rate, 1.0)
        self.assertEqual(result.max_drawdown, 0.0)
        self.assertEqual(len(result.equity_curve), 4)
        self.assertEqual(result.trades[0]["symbol"], "EURUSD")
        self.assertEqual(result.trades[0]["entry_price"], 1.083)
        self.assertEqual(result.trades[0]["exit_price"], 1.085)

    def test_multi_symbol_positions_are_closed_independently(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("AAA", AssetClass.ETF, 10.0, base),
            MarketEvent("BBB", AssetClass.ETF, 20.0, base + timedelta(minutes=1)),
            MarketEvent("AAA", AssetClass.ETF, 11.0, base + timedelta(minutes=2)),
            MarketEvent("BBB", AssetClass.ETF, 21.0, base + timedelta(minutes=3)),
            MarketEvent("BBB", AssetClass.ETF, 18.0, base + timedelta(minutes=4)),
            MarketEvent("AAA", AssetClass.ETF, 12.0, base + timedelta(minutes=5)),
        ]

        result = BacktestEngine(MomentumStrategy(1, 1)).run(events)

        self.assertEqual(result.trade_count, 2)
        self.assertEqual({trade["symbol"] for trade in result.trades}, {"AAA", "BBB"})
        self.assertEqual(result.total_return, -2.0)
        self.assertEqual(result.win_rate, 0.5)

    def test_trade_quantity_scales_return(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("510300", AssetClass.ETF, price, base + timedelta(days=i))
            for i, price in enumerate((3.5, 3.6, 3.8, 4.0))
        ]

        result = BacktestEngine(MomentumStrategy(2, 100)).run(events)

        self.assertAlmostEqual(result.total_return, 20.0)
        self.assertEqual(result.trades[0]["quantity"], 100)

    def test_commission_rate_reduces_net_return(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
            for i, price in enumerate((1.0, 1.1, 1.2))
        ]

        result = BacktestEngine(MomentumStrategy(1, 10), commission_rate=0.01).run(events)

        self.assertAlmostEqual(result.total_return, 0.77)

    def test_slippage_rate_reduces_long_return(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
            for i, price in enumerate((1.0, 1.1, 1.2))
        ]

        result = BacktestEngine(MomentumStrategy(1, 10), slippage_rate=0.01).run(events)

        self.assertAlmostEqual(result.total_return, 0.77)

    def test_trade_records_cost_breakdown(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
            for i, price in enumerate((1.0, 1.1, 1.2))
        ]

        result = BacktestEngine(MomentumStrategy(1, 10), commission_rate=0.01, slippage_rate=0.01).run(events)

        trade = result.trades[0]
        self.assertGreater(trade["commission_cost"], 0)
        self.assertGreater(trade["slippage_cost"], 0)
        self.assertAlmostEqual(trade["return"], trade["gross_return"] - trade["commission_cost"] - trade["slippage_cost"])

    def test_chronological_split_has_no_overlap(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [MarketEvent("EURUSD", AssetClass.FX, 1 + i / 100, base + timedelta(days=i)) for i in range(10)]

        in_sample, out_of_sample = chronological_split(events, 0.7)

        self.assertEqual(len(in_sample), 7)
        self.assertEqual(len(out_of_sample), 3)
        self.assertLess(in_sample[-1].timestamp, out_of_sample[0].timestamp)

    def test_backtest_rejects_out_of_order_events(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, 1.08, base + timedelta(minutes=1)),
            MarketEvent("EURUSD", AssetClass.FX, 1.09, base),
        ]

        with self.assertRaises(ValueError):
            BacktestEngine(MomentumStrategy(1, 1)).run(events)

    def test_reusing_strategy_for_backtest_is_deterministic(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
            for i, price in enumerate((1.08, 1.081, 1.083))
        ]
        strategy = MomentumStrategy(2, 1)

        first = BacktestEngine(strategy).run(events)
        second = BacktestEngine(strategy).run(events)

        self.assertEqual(first.total_return, second.total_return)
        self.assertEqual(first.trade_count, second.trade_count)

    def test_result_can_be_serialized_for_research_consumers(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
            for i, price in enumerate((1.08, 1.081, 1.083))
        ]

        payload = BacktestEngine(MomentumStrategy(2, 1)).run(events).to_dict()

        self.assertEqual(payload["trade_count"], 1)
        self.assertEqual(payload["trades"][0]["symbol"], "EURUSD")
        self.assertIsInstance(payload["equity_curve"], list)
        self.assertIsNone(payload["profit_factor"])
        self.assertIn("total_commission", payload)
        self.assertIn("total_slippage", payload)
        self.assertIn("max_consecutive_losses", payload)

    def test_sell_signal_closes_position_at_signal_price(self):
        class EntryExitStrategy:
            def reset(self):
                self.index = 0

            def on_event(self, event):
                self.index += 1
                if self.index == 1:
                    return Signal(event.symbol, OrderSide.BUY, 10, "entry")
                if self.index == 3:
                    return Signal(event.symbol, OrderSide.SELL, 10, "exit")
                return None

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, base + timedelta(days=i))
            for i, price in enumerate((1.0, 1.1, 1.2, 1.3))
        ]

        result = BacktestEngine(EntryExitStrategy()).run(events)

        self.assertEqual(result.trade_count, 1)
        self.assertEqual(result.trades[0]["exit_price"], 1.2)
        self.assertAlmostEqual(result.total_return, 2.0)
        self.assertAlmostEqual(result.equity_curve[-1], 2.0)

    def test_event_backtest_filters_events_outside_market_session(self):
        class AlwaysBuy:
            def reset(self):
                pass

            def on_event(self, event):
                return Signal(event.symbol, OrderSide.BUY, 100, "entry")

        from zoneinfo import ZoneInfo
        china = ZoneInfo("Asia/Shanghai")
        events = [
            MarketEvent("510300", AssetClass.ETF, 4.0, datetime(2026, 1, 5, 8, 0, tzinfo=china)),
            MarketEvent("510300", AssetClass.ETF, 4.1, datetime(2026, 1, 5, 9, 31, tzinfo=china)),
        ]

        result = BacktestEngine(AlwaysBuy(), execution_policy=MarketExecutionPolicy()).run(events)

        self.assertEqual(result.signal_count, 1)
        self.assertEqual(result.restriction_count, 1)
        self.assertEqual(result.restrictions[0]["reason"], "market_closed")

    def test_event_backtest_blocks_limit_up_buy(self):
        class AlwaysBuy:
            def reset(self):
                pass

            def on_event(self, event):
                return Signal(event.symbol, OrderSide.BUY, 100, "entry")

        from zoneinfo import ZoneInfo
        moment = datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        policy = MarketExecutionPolicy(
            previous_closes={"510300": 4.0},
            price_limit_percentages={"510300": 0.10},
        )

        result = BacktestEngine(AlwaysBuy(), execution_policy=policy).run([
            MarketEvent("510300", AssetClass.ETF, 4.4, moment),
        ])

        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.restrictions[0]["reason"], "buy_blocked_at_upper_price_limit")

    def test_event_backtest_blocks_expired_futures_contract_after_rollover(self):
        class AlwaysBuy:
            def reset(self):
                pass

            def on_event(self, event):
                return Signal(event.symbol, OrderSide.BUY, 1, "entry")

        from zoneinfo import ZoneInfo
        china = ZoneInfo("Asia/Shanghai")
        rollover = datetime(2026, 1, 15, 9, 0, tzinfo=china)
        policy = MarketExecutionPolicy(rollover_rules=[
            FuturesRolloverRule("IF2601", "IF2602", rollover),
        ])

        result = BacktestEngine(AlwaysBuy(), execution_policy=policy).run([
            MarketEvent("IF2601", AssetClass.FUTURES, 4000, rollover + timedelta(minutes=1)),
        ])

        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.restrictions[0]["reason"], "expired_contract_use_IF2602")
        self.assertFalse(result.to_dict()["execution_enabled"])

    def test_event_backtest_can_roll_open_futures_position_to_successor(self):
        class AlwaysBuy:
            def reset(self):
                pass

            def on_event(self, event):
                return Signal(event.symbol, OrderSide.BUY, 2, "entry")

        from zoneinfo import ZoneInfo
        china = ZoneInfo("Asia/Shanghai")
        rollover_at = datetime(2026, 1, 15, 9, 0, tzinfo=china)
        policy = MarketExecutionPolicy(
            rollover_rules=[FuturesRolloverRule("IF2601", "IF2602", rollover_at)],
            auto_rollover=True,
        )
        events = [
            MarketEvent("IF2601", AssetClass.FUTURES, 4000, rollover_at - timedelta(days=1)),
            MarketEvent("IF2602", AssetClass.FUTURES, 4010, rollover_at + timedelta(hours=1)),
        ]

        result = BacktestEngine(
            AlwaysBuy(), commission_rate=0.001, slippage_rate=0.001, execution_policy=policy
        ).run(events)
        payload = result.to_dict()

        self.assertEqual(payload["rollover_count"], 1)
        self.assertEqual(payload["rollovers"][0]["from_symbol"], "IF2601")
        self.assertEqual(payload["rollovers"][0]["to_symbol"], "IF2602")
        self.assertEqual(payload["rollovers"][0]["quantity"], 2)
        self.assertEqual(payload["trades"][0]["exit_reason"], "rollover")
        self.assertGreater(payload["total_commission"], 0)
        self.assertGreater(payload["total_slippage"], 0)
