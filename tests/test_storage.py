import unittest
import sqlite3
import stat
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from quant_platform.domain import AssetClass, OrderIntent, OrderSide
from quant_platform.storage import SimulationAccountStore, SqliteAuditStore


class StorageTests(unittest.TestCase):
    def test_store_records_and_reads_order_audit_event(self):
        store = SqliteAuditStore(":memory:")
        intent = OrderIntent(symbol="EURUSD", side=OrderSide.BUY, quantity=0.01, strategy_id="demo")

        store.record_order(order_id="order-1", intent=intent, status="pending")
        records = store.list_orders()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["order_id"], "order-1")
        self.assertEqual(records[0]["symbol"], "EURUSD")
        self.assertEqual(records[0]["status"], "pending")

    def test_store_updates_order_status(self):
        store = SqliteAuditStore(":memory:")
        intent = OrderIntent(symbol="510300", side=OrderSide.BUY, quantity=100, strategy_id="demo")
        store.record_order(order_id="order-2", intent=intent, status="pending")

        store.update_status("order-2", "filled")

        self.assertEqual(store.list_orders()[0]["status"], "filled")

    def test_store_keeps_append_only_status_history(self):
        store = SqliteAuditStore(":memory:")
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo")
        store.record_order("order-history", intent, "pending")
        store.update_status("order-history", "accepted")
        store.update_status("order-history", "filled")

        events = store.list_audit_events("order-history")

        self.assertEqual([event["status"] for event in events], ["pending", "accepted", "filled"])
        self.assertEqual(len(events), 3)
        self.assertTrue(all(event["created_at"].endswith("+00:00") for event in events))

    def test_store_preserves_market_context(self):
        store = SqliteAuditStore(":memory:")
        intent = OrderIntent(
            "510300", OrderSide.BUY, 100, "demo", asset_class=AssetClass.ETF, reference_price=3.5
        )

        store.record_order("order-ctx", intent, "accepted")

        order = store.list_orders()[0]
        self.assertEqual(order["asset_class"], "etf")
        self.assertEqual(order["reference_price"], 3.5)

    def test_store_preserves_client_order_id_for_idempotency_audit(self):
        store = SqliteAuditStore(":memory:")
        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo", client_order_id="client-1")

        store.record_order("order-client", intent, "accepted")

        self.assertEqual(store.list_orders()[0]["client_order_id"], "client-1")

    def test_store_migrates_legacy_orders_table(self):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE orders (order_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL, quantity REAL NOT NULL, strategy_id TEXT NOT NULL, status TEXT NOT NULL)"
        )
        connection.commit()
        store = SqliteAuditStore(connection=connection)

        intent = OrderIntent("EURUSD", OrderSide.BUY, 0.01, "legacy", reference_price=1.08)
        store.record_order("legacy-1", intent, "accepted")

        self.assertEqual(store.list_orders()[0]["reference_price"], 1.08)

    def test_simulation_store_lists_account_snapshots_for_acceptance_evidence(self):
        with TemporaryDirectory() as directory:
            store = SimulationAccountStore(Path(directory) / "simulation.db")
            store.save_snapshot("paper-a", 9_000, 10_000, {"510300.SH": 100})
            store.save_snapshot("paper-b", 8_000, 9_000, {"510500.SH": 100})

            snapshots = store.list_snapshots("paper-a")

            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0]["account_id"], "paper-a")
            self.assertEqual(snapshots[0]["positions"], {"510300.SH": 100})

    def test_sqlite_runtime_database_is_owner_read_write_only(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "platform.db"
            SqliteAuditStore(str(database))

            permissions = stat.S_IMODE(database.stat().st_mode)

            self.assertEqual(permissions, 0o600)

    def test_store_is_usable_across_threads(self):
        with TemporaryDirectory() as directory:
            audit = SqliteAuditStore(str(Path(directory) / "audit.db"))
            simulation = SimulationAccountStore(str(Path(directory) / "sim.db"))
            errors = []

            def worker():
                try:
                    intent = OrderIntent("510300", OrderSide.BUY, 100, "demo", asset_class=AssetClass.ETF)
                    audit.record_order("threaded-1", intent, "accepted")
                    simulation.save_session_state("paper-a", {"version": 1, "account_id": "paper-a"})
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(audit.list_orders()[0]["order_id"], "threaded-1")
            self.assertEqual(simulation.load_session_state("paper-a")["account_id"], "paper-a")
