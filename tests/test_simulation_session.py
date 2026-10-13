import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.risk import OperationalRiskGuard, RiskLimits
from quant_platform.simulation import PaperSimulationRegistry, PaperSimulationSession
from quant_platform.storage import SimulationAccountStore


class PaperSimulationSessionTests(unittest.TestCase):
    def test_market_event_updates_fills_account_equity_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SimulationAccountStore(Path(directory) / "simulation.db")
            session = PaperSimulationSession(
                account_id="paper-main",
                initial_cash=1_000,
                account_store=store,
                max_participation_rate=0.5,
            )
            order_id = session.submit(
                OrderIntent("510300.SH", OrderSide.BUY, 100, "cta", reference_price=4.0)
            )

            first = session.on_market("510300.SH", price=4.0, volume=100)
            second = session.on_market("510300.SH", price=5.0, volume=100)

            self.assertEqual(first["fills"][0]["quantity"], 50)
            self.assertEqual(second["fills"][0]["quantity"], 50)
            self.assertEqual(second["orders"][order_id], "filled")
            self.assertEqual(second["account"]["cash"], 550)
            self.assertEqual(second["account"]["positions"], {"510300.SH": 100})
            self.assertEqual(second["account"]["equity"], 1_050)
            snapshot = store.latest_snapshot("paper-main")
            self.assertEqual(snapshot["equity"], 1_050)
            self.assertEqual(snapshot["positions"], {"510300.SH": 100})
            self.assertTrue(second["paper_only"])
            self.assertFalse(second["execution_enabled"])

    def test_account_drawdown_halts_new_simulation_orders(self):
        session = PaperSimulationSession(
            account_id="paper-risk",
            initial_cash=1_000,
            risk_limits=RiskLimits(max_drawdown=200),
            max_participation_rate=1.0,
        )
        session.submit(OrderIntent("510300.SH", OrderSide.BUY, 100, "cta", reference_price=4.0))
        session.on_market("510300.SH", price=4.0, volume=100)

        state = session.on_market("510300.SH", price=1.0, volume=0)

        self.assertEqual(state["risk"]["risk_level"], "halt")
        self.assertIn("drawdown_limit", state["risk"]["alerts"])
        with self.assertRaisesRegex(RuntimeError, "simulation_halted"):
            session.submit(OrderIntent("510300.SH", OrderSide.BUY, 1, "cta", reference_price=1.0))

    def test_operational_guard_is_enforced_by_simulation_session(self):
        guard = OperationalRiskGuard(max_consecutive_losses=2, max_api_failures=2)
        session = PaperSimulationSession(
            account_id="paper-ops",
            initial_cash=1_000,
            operational_guard=guard,
        )
        session.record_trade_pnl(-10)
        session.record_trade_pnl(-5)

        state = session.state()

        self.assertTrue(state["operational_risk"]["halted"])
        with self.assertRaisesRegex(RuntimeError, "consecutive_loss_limit"):
            session.submit(OrderIntent("EURUSD", OrderSide.BUY, 0.01, "cta", reference_price=1.1))

    def test_registry_restores_partial_orders_and_risk_state_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SimulationAccountStore(Path(directory) / "simulation.db")
            first_registry = PaperSimulationRegistry(store)
            first = first_registry.create(
                account_id="paper-recovery",
                initial_cash=1_000,
                max_consecutive_losses=2,
                max_participation_rate=0.5,
            )
            order_id = first.submit(
                OrderIntent("510300.SH", OrderSide.BUY, 100, "cta", reference_price=4.0)
            )
            first.on_market("510300.SH", price=4.0, volume=100)
            first.record_trade_pnl(-10)
            first.activate_kill_switch("operator_requested")

            restored_registry = PaperSimulationRegistry(store)
            restored = restored_registry.get("paper-recovery")
            restored_state = restored.state()

            self.assertEqual(restored_state["orders"][order_id], "partially_filled")
            self.assertEqual(restored_state["account"]["positions"], {"510300.SH": 50})
            self.assertEqual(restored_state["operational_risk"]["consecutive_losses"], 1)
            self.assertIn("operator_requested", restored_state["operational_risk"]["alerts"])


class OperationalRiskGuardTests(unittest.TestCase):
    def test_consecutive_losses_trigger_halt_and_profit_resets_counter(self):
        guard = OperationalRiskGuard(max_consecutive_losses=3)
        guard.record_trade_pnl(-10)
        guard.record_trade_pnl(-5)
        guard.record_trade_pnl(2)
        guard.record_trade_pnl(-4)
        guard.record_trade_pnl(-3)
        self.assertFalse(guard.status()["halted"])

        guard.record_trade_pnl(-2)
        state = guard.status()

        self.assertTrue(state["halted"])
        self.assertIn("consecutive_loss_limit", state["alerts"])
        self.assertEqual(state["consecutive_losses"], 3)
        self.assertFalse(state["execution_enabled"])

    def test_stale_market_data_and_api_failures_trigger_halt(self):
        now = datetime(2026, 10, 10, 1, 0, tzinfo=timezone.utc)
        guard = OperationalRiskGuard(
            market_data_timeout_seconds=30,
            max_api_failures=2,
        )
        guard.record_market_event(now - timedelta(seconds=31))
        guard.record_api_result(success=False)
        guard.record_api_result(success=False)

        state = guard.status(now=now)

        self.assertTrue(state["halted"])
        self.assertIn("market_data_stale", state["alerts"])
        self.assertIn("api_failure_limit", state["alerts"])

    def test_manual_kill_switch_is_explicit_and_irreversible_without_reset(self):
        guard = OperationalRiskGuard()

        guard.activate_kill_switch("operator_requested")
        state = guard.status()

        self.assertTrue(state["halted"])
        self.assertIn("operator_requested", state["alerts"])


if __name__ == "__main__":
    unittest.main()
