from datetime import datetime, timedelta, timezone
import unittest

from quant_platform.domain import AssetClass, OrderIntent, OrderSide
from quant_platform.simulation import PaperSimulationSession
from quant_platform.simulation_acceptance import (
    SimulationAcceptanceGate,
    reconcile_simulation_snapshot,
)


class SimulationAcceptanceTests(unittest.TestCase):
    def _filled_session(self):
        session = PaperSimulationSession("paper-acceptance", 10_000)
        order_id = session.submit(OrderIntent(
            symbol="510300.SH",
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="acceptance-test",
            asset_class=AssetClass.ETF,
            reference_price=10,
        ))
        session.on_market("510300.SH", 10, 1_000)
        return session, order_id

    def test_reconciliation_matches_cash_positions_fills_and_order_state(self):
        session, order_id = self._filled_session()

        report = reconcile_simulation_snapshot(session.snapshot())

        self.assertTrue(report["passed"])
        self.assertEqual(report["fill_count"], 1)
        self.assertEqual(report["checks"]["cash"]["difference"], 0.0)
        self.assertEqual(report["checks"]["positions"]["differences"], {})
        self.assertEqual(report["checks"]["orders"]["mismatches"], {})
        self.assertEqual(report["filled_quantities"][order_id], 100.0)
        self.assertFalse(report["execution_enabled"])

    def test_reconciliation_detects_tampered_cash(self):
        session, _order_id = self._filled_session()
        snapshot = session.snapshot()
        snapshot["matcher"]["cash"] += 10

        report = reconcile_simulation_snapshot(snapshot)

        self.assertFalse(report["passed"])
        self.assertEqual(report["checks"]["cash"]["difference"], 10.0)
        self.assertIn("cash_mismatch", report["discrepancies"])

    def test_acceptance_requires_30_distinct_days_and_reconciliation(self):
        session, _order_id = self._filled_session()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        daily_snapshots = [
            {"created_at": (start + timedelta(days=offset)).isoformat()}
            for offset in range(30)
        ]

        report = SimulationAcceptanceGate(30).assess(session.snapshot(), daily_snapshots)

        self.assertTrue(report["eligible_for_manual_review"])
        self.assertEqual(report["paper_trading_days"], 30)
        self.assertEqual(report["status"], "eligible_for_manual_review")
        self.assertFalse(report["live_trading_enabled"])
        self.assertFalse(report["execution_enabled"])

    def test_acceptance_reports_days_and_risk_blockers(self):
        session, _order_id = self._filled_session()
        snapshot = session.snapshot()
        snapshot["halted"] = True
        daily_snapshots = [
            {"created_at": f"2026-01-{day:02d}T09:30:00+00:00"}
            for day in range(1, 30)
        ]

        report = SimulationAcceptanceGate(30).assess(snapshot, daily_snapshots)

        self.assertFalse(report["eligible_for_manual_review"])
        self.assertIn("minimum_paper_trading_days_not_met", report["blockers"])
        self.assertIn("account_risk_halted", report["blockers"])


if __name__ == "__main__":
    unittest.main()
