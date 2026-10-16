import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_platform.domain import AssetClass, OrderIntent, OrderSide
from quant_platform.http_api import simulation_track_rebalance_payload
from quant_platform.run_history import RunHistoryStore
from quant_platform.simulation import PaperSimulationRegistry
from quant_platform.storage import SimulationAccountStore


class TrackRebalanceTests(unittest.TestCase):
    def test_track_rebalance_marks_and_rebalances(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        session = registry.create("p9-run-1", initial_cash=1_000_000.0)
        # initial position: 50000 shares of A at 10 = 500000 (50%)
        session.submit(OrderIntent("A", OrderSide.BUY, 50000.0, "test", AssetClass.ETF))
        session.on_market("A", 10.0, 1e9)

        store = RunHistoryStore(str(Path(tempfile.mkdtemp()) / "h.json"))
        store.add({
            "kind": "phase9", "run_id": "p9-run-1", "asset_class": "etf",
            "targets": [{"symbol": "A", "weight": 0.5}, {"symbol": "B", "weight": 0.5}],
        })

        with patch("quant_platform.http_api.get_store", return_value=store), \
             patch("quant_platform.http_api._latest_close_for_fill", return_value=10.0):
            result = simulation_track_rebalance_payload({"account_id": "p9-run-1"}, registry)

        self.assertIn("rebalanced", result)
        self.assertIn("target_weights", result)
        self.assertEqual(set(result["target_weights"]), {"A", "B"})
        self.assertFalse(result["execution_enabled"])
        self.assertIn("B", session.matcher.positions)

    def test_track_rebalance_requires_record(self):
        registry = PaperSimulationRegistry(SimulationAccountStore(":memory:"))
        registry.create("no-run", initial_cash=1_000_000.0)
        store = RunHistoryStore(str(Path(tempfile.mkdtemp()) / "h.json"))
        with patch("quant_platform.http_api.get_store", return_value=store):
            with self.assertRaisesRegex(ValueError, "未找到"):
                simulation_track_rebalance_payload({"account_id": "no-run"}, registry)


if __name__ == "__main__":
    unittest.main()
