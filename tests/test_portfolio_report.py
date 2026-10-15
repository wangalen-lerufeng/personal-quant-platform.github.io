import tempfile
import unittest
from pathlib import Path

from quant_platform.portfolio_report import _cost_basis, build_account_report
from quant_platform.simulation import PaperSimulationRegistry
from quant_platform.storage import SimulationAccountStore


class FakeFill:
    def __init__(self, symbol, side, quantity, price):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price


class FakeMatcher:
    def __init__(self):
        self.positions = {"600519.SH": 100.0}
        self.cash = 5000.0
        self.fills = [FakeFill("600519.SH", "buy", 100.0, 10.0)]
        self._pending = {}


class FakeSession:
    account_id = "test-acc"
    latest_prices = {"600519.SH": 12.0}
    matcher = FakeMatcher()


class CostBasisTests(unittest.TestCase):
    def test_buy_only_average(self):
        fills = [FakeFill("A", "buy", 100, 10.0), FakeFill("A", "buy", 100, 20.0)]
        self.assertEqual(_cost_basis(fills), {"A": 15.0})

    def test_sell_reduces_at_average(self):
        fills = [FakeFill("A", "buy", 100, 10.0), FakeFill("A", "sell", 50, 12.0)]
        self.assertEqual(_cost_basis(fills), {"A": 10.0})


class AccountReportTests(unittest.TestCase):
    def test_build_account_report(self):
        report = build_account_report(FakeSession())
        self.assertEqual(report["account_id"], "test-acc")
        self.assertEqual(report["cash"], 5000.0)
        self.assertEqual(report["market_value"], 1200.0)
        self.assertEqual(report["cost_value"], 1000.0)
        self.assertEqual(report["unrealized_pnl"], 200.0)
        self.assertAlmostEqual(report["unrealized_return"], 0.2)
        self.assertEqual(len(report["holdings"]), 1)
        h = report["holdings"][0]
        self.assertEqual(h["symbol"], "600519.SH")
        self.assertEqual(h["cost"], 10.0)
        self.assertEqual(h["latest"], 12.0)
        self.assertAlmostEqual(h["return"], 0.2)


class DeleteAccountTests(unittest.TestCase):
    def test_registry_delete_removes_account(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        registry.create("paper-main", initial_cash=1000.0)
        self.assertIn("paper-main", [s["account_id"] for s in registry.list_states()])
        result = registry.delete("paper-main")
        self.assertEqual(result["deleted"], "paper-main")
        self.assertEqual(registry.list_states(), [])
        with self.assertRaises(KeyError):
            registry.get("paper-main")

    def test_store_delete_account_removes_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SimulationAccountStore(Path(directory) / "s.db")
            store.save_session_state("paper-main", {"account_id": "paper-main"})
            store.save_snapshot("paper-main", 9000, 10000, {"A": 1})
            store.delete_account("paper-main")
            self.assertEqual(store.list_session_states(), [])
            self.assertIsNone(store.latest_snapshot("paper-main"))


if __name__ == "__main__":
    unittest.main()
