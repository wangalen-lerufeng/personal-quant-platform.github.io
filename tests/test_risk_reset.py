import unittest

from quant_platform.risk import RiskEngine, RiskLimits


class RiskResetTests(unittest.TestCase):
    def test_new_trading_day_resets_daily_loss_but_keeps_drawdown(self):
        engine = RiskEngine(RiskLimits(max_daily_loss=500, max_drawdown=1000))
        engine.record_pnl(-500)
        engine.record_drawdown(700)

        engine.start_new_day()

        self.assertEqual(engine.daily_loss, 0.0)
        self.assertEqual(engine.drawdown, 700)
