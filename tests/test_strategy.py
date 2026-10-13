import unittest

from quant_platform.domain import AssetClass, MarketEvent, OrderSide
from quant_platform.strategy import MomentumStrategy


class StrategyTests(unittest.TestCase):
    def test_strategy_emits_buy_after_upward_breakout(self):
        strategy = MomentumStrategy(lookback=3, quantity=0.01)
        prices = [1.08, 1.081, 1.082, 1.084]

        signals = [strategy.on_event(MarketEvent("EURUSD", AssetClass.FX, price, strategy.timestamp(i))) for i, price in enumerate(prices)]

        self.assertIsNone(signals[0])
        self.assertIsNone(signals[1])
        self.assertIsNone(signals[2])
        self.assertEqual(signals[3].side, OrderSide.BUY)

    def test_strategy_ignores_other_symbols(self):
        strategy = MomentumStrategy(lookback=2, quantity=100)
        event = MarketEvent("510300", AssetClass.ETF, 3.5, strategy.timestamp(0))

        self.assertIsNone(strategy.on_event(event))

    def test_strategy_does_not_repeat_entry_signal_without_reset(self):
        strategy = MomentumStrategy(lookback=2, quantity=0.01)
        prices = [1.08, 1.081, 1.083, 1.085]

        signals = [strategy.on_event(MarketEvent("EURUSD", AssetClass.FX, price, strategy.timestamp(i))) for i, price in enumerate(prices)]

        self.assertEqual(sum(signal is not None for signal in signals), 1)

    def test_strategy_can_reset_symbol_for_a_new_trade_cycle(self):
        strategy = MomentumStrategy(lookback=2, quantity=0.01)
        for i, price in enumerate((1.08, 1.081, 1.083)):
            strategy.on_event(MarketEvent("EURUSD", AssetClass.FX, price, strategy.timestamp(i)))

        strategy.reset("EURUSD")
        signal = strategy.on_event(MarketEvent("EURUSD", AssetClass.FX, 1.085, strategy.timestamp(3)))

        self.assertIsNotNone(signal)

    def test_research_mode_can_exit_on_negative_momentum(self):
        strategy = MomentumStrategy(lookback=2, quantity=0.01, exit_on_negative=True)
        prices = [1.08, 1.081, 1.083, 1.085, 1.08, 1.07]
        signals = [strategy.on_event(MarketEvent("EURUSD", AssetClass.FX, price, strategy.timestamp(i))) for i, price in enumerate(prices)]
        self.assertEqual(signals[2].side, OrderSide.BUY)
        self.assertEqual(signals[4].side, OrderSide.SELL)
