import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Any

from .domain import OrderIntent


def _secure_database_file(database: str) -> None:
    if database != ":memory:":
        os.chmod(database, 0o600)


class SqliteAuditStore:
    """Local development store; production can implement the same methods on PostgreSQL."""

    def __init__(self, database: str = ":memory:", connection=None):
        # ThreadingHTTPServer serves each request in a worker thread, so the
        # connection created at startup is used from a different thread.
        self.connection = connection or sqlite3.connect(database, check_same_thread=False)
        if connection is None:
            _secure_database_file(database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                strategy_id TEXT NOT NULL,
                status TEXT NOT NULL,
                asset_class TEXT,
                reference_price REAL,
                client_order_id TEXT
            )
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(orders)")}
        if "asset_class" not in columns:
            self.connection.execute("ALTER TABLE orders ADD COLUMN asset_class TEXT")
        if "reference_price" not in columns:
            self.connection.execute("ALTER TABLE orders ADD COLUMN reference_price REAL")
        if "client_order_id" not in columns:
            self.connection.execute("ALTER TABLE orders ADD COLUMN client_order_id TEXT")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def record_order(self, order_id: str, intent: OrderIntent, status: str) -> None:
        self.connection.execute(
            "INSERT INTO orders(order_id, symbol, side, quantity, strategy_id, status, asset_class, reference_price, client_order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order_id,
                intent.symbol,
                intent.side.value,
                intent.quantity,
                intent.strategy_id,
                status,
                intent.asset_class.value if intent.asset_class is not None else None,
                intent.reference_price,
                intent.client_order_id,
            ),
        )
        self.connection.execute(
            "INSERT INTO audit_events(order_id, event_type, status, created_at) VALUES (?, ?, ?, ?)",
            (order_id, "order_recorded", status, datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()

    def update_status(self, order_id: str, status: str) -> None:
        cursor = self.connection.execute("UPDATE orders SET status = ? WHERE order_id = ?", (status, order_id))
        if cursor.rowcount != 1:
            raise KeyError(f"unknown order: {order_id}")
        self.connection.execute(
            "INSERT INTO audit_events(order_id, event_type, status, created_at) VALUES (?, ?, ?, ?)",
            (order_id, "order_status_changed", status, datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()

    def list_orders(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM orders ORDER BY rowid").fetchall()
        return [dict(row) for row in rows]

    def find_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM orders WHERE client_order_id = ? ORDER BY rowid LIMIT 1",
            (client_order_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def list_audit_events(self, order_id: str | None = None) -> list[dict[str, Any]]:
        if order_id is None:
            rows = self.connection.execute("SELECT * FROM audit_events ORDER BY event_id").fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM audit_events WHERE order_id = ? ORDER BY event_id", (order_id,)
            ).fetchall()
        return [dict(row) for row in rows]


class SimulationAccountStore:
    """Append-only snapshots for recoverable paper-account state."""

    def __init__(self, database: str):
        # See SqliteAuditStore: the connection is shared across request threads.
        self.connection = sqlite3.connect(str(database), check_same_thread=False)
        _secure_database_file(str(database))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_account_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                positions_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_session_states (
                account_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def save_snapshot(self, account_id: str, cash: float, equity: float, positions: dict[str, float]) -> int:
        if not account_id.strip():
            raise ValueError("account_id is required")
        if cash < 0 or equity < 0:
            raise ValueError("cash and equity must be non-negative")
        cursor = self.connection.execute(
            "INSERT INTO simulation_account_snapshots(account_id, cash, equity, positions_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (account_id, cash, equity, json.dumps(positions, ensure_ascii=False, sort_keys=True), datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def latest_snapshot(self, account_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM simulation_account_snapshots WHERE account_id = ? ORDER BY snapshot_id DESC LIMIT 1",
            (account_id,),
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["positions"] = json.loads(payload.pop("positions_json"))
        return payload

    def list_snapshots(self, account_id: str) -> list[dict[str, Any]]:
        if not account_id.strip():
            raise ValueError("account_id is required")
        rows = self.connection.execute(
            "SELECT * FROM simulation_account_snapshots WHERE account_id = ? ORDER BY snapshot_id",
            (account_id,),
        ).fetchall()
        snapshots = []
        for row in rows:
            payload = dict(row)
            payload["positions"] = json.loads(payload.pop("positions_json"))
            snapshots.append(payload)
        return snapshots

    def save_session_state(self, account_id: str, state: dict[str, Any]) -> None:
        if not account_id.strip():
            raise ValueError("account_id is required")
        self.connection.execute(
            """
            INSERT INTO simulation_session_states(account_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (
                account_id,
                json.dumps(state, ensure_ascii=False, sort_keys=True),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    def load_session_state(self, account_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT state_json FROM simulation_session_states WHERE account_id = ?",
            (account_id,),
        ).fetchone()
        return None if row is None else json.loads(row["state_json"])

    def list_session_states(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT state_json FROM simulation_session_states ORDER BY account_id"
        ).fetchall()
        return [json.loads(row["state_json"]) for row in rows]

    def delete_account(self, account_id: str) -> None:
        if not account_id.strip():
            raise ValueError("account_id is required")
        self.connection.execute(
            "DELETE FROM simulation_session_states WHERE account_id = ?", (account_id,)
        )
        self.connection.execute(
            "DELETE FROM simulation_account_snapshots WHERE account_id = ?", (account_id,)
        )
        self.connection.commit()
