import tempfile
import unittest
from pathlib import Path

from quant_platform.risk import RiskMonitor, RiskLimits
from quant_platform.storage import SimulationAccountStore


class SimulationAccountTests(unittest.TestCase):
    def test_account_snapshot_persists_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SimulationAccountStore(Path(directory) / "simulation.db")
            store.save_snapshot("paper-main", cash=9000, equity=10200, positions={"510300.SH": 300})

            snapshot = store.latest_snapshot("paper-main")

            self.assertEqual(snapshot["cash"], 9000)
            self.assertEqual(snapshot["equity"], 10200)
            self.assertEqual(snapshot["positions"]["510300.SH"], 300)

    def test_risk_monitor_reports_drawdown_and_daily_loss(self):
        monitor = RiskMonitor(RiskLimits(max_daily_loss=500, max_drawdown=1000), initial_equity=10000)
        monitor.observe(equity=10500)
        state = monitor.observe(equity=9400)

        self.assertEqual(state["drawdown"], 1100)
        self.assertEqual(state["daily_loss"], 600)
        self.assertEqual(state["risk_level"], "halt")
        self.assertIn("drawdown_limit", state["alerts"])
        self.assertIn("daily_loss_limit", state["alerts"])

    def test_simulation_session_state_round_trips_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SimulationAccountStore(Path(directory) / "simulation.db")
            state = {
                "version": 1,
                "account_id": "paper-main",
                "matcher": {"cash": 9000, "positions": {"510300.SH": 100}},
            }

            store.save_session_state("paper-main", state)

            self.assertEqual(store.load_session_state("paper-main"), state)
            self.assertEqual(store.list_session_states(), [state])
