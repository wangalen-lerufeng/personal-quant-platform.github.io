from datetime import datetime, timedelta, timezone
import unittest

from quant_platform.backtest import BacktestEngine
from quant_platform.config import RuntimeConfig
from quant_platform.control import ControlPlane
from quant_platform.domain import AssetClass, MarketEvent
from quant_platform.experiments import ExperimentStore
from quant_platform.research_tools import ResearchTools
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.storage import SqliteAuditStore
from quant_platform.strategy import MomentumStrategy


class ResearchBacktestTests(unittest.TestCase):
    def test_research_tool_runs_backtest_and_saves_experiment(self):
        control = ControlPlane(RuntimeConfig(), RiskEngine(RiskLimits()), SqliteAuditStore(":memory:"))
        experiments = ExperimentStore()
        tools = ResearchTools(control, experiments)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, start + timedelta(days=i))
            for i, price in enumerate((1.08, 1.081, 1.083, 1.085))
        ]

        result = tools.run_backtest("eurusd-demo", MomentumStrategy(2, 0.01), events)

        self.assertEqual(result.trade_count, 1)
        self.assertEqual(tools.experiment_results()[0]["strategy_id"], "eurusd-demo")
        self.assertEqual(tools.experiment_results()[0]["metrics"]["win_rate"], 1.0)
        self.assertEqual(len(tools.experiment_results()[0]["metrics"]["trades"]), 1)

    def test_research_tool_records_backtest_cost_assumption(self):
        control = ControlPlane(RuntimeConfig(), RiskEngine(RiskLimits()), SqliteAuditStore(":memory:"))
        experiments = ExperimentStore()
        tools = ResearchTools(control, experiments)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            MarketEvent("EURUSD", AssetClass.FX, price, start + timedelta(days=i))
            for i, price in enumerate((1.0, 1.1, 1.2))
        ]

        result = tools.run_backtest("costed-demo", MomentumStrategy(1, 10), events, commission_rate=0.01, slippage_rate=0.01)

        self.assertAlmostEqual(result.total_return, 0.5401)
        self.assertEqual(tools.experiment_results()[0]["metrics"]["commission_rate"], 0.01)
        self.assertEqual(tools.experiment_results()[0]["metrics"]["slippage_rate"], 0.01)
