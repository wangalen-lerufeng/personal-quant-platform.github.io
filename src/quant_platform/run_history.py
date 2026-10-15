"""Append-only run history for post-hoc review (复盘).

Persists each phase9 portfolio-research run and each forward-validation run to
a local JSON file so past results can be browsed and correlated with the paper
simulation. Read-only with respect to trading.
"""

import json
from datetime import datetime
from pathlib import Path


class RunHistoryStore:
    """Store run records keyed by ``run_id`` in a JSON file."""

    def __init__(self, store_path: str = "data/run_history.json"):
        self.store_path = Path(store_path)

    def add(self, record: dict) -> str:
        """Insert (or replace) a record, returning its run_id."""
        record = dict(record)
        run_id = record.get("run_id") or self._new_id(record.get("kind", "run"))
        record["run_id"] = run_id
        record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        state = self._load()
        state["runs"] = [
            r for r in state.get("runs", []) if r.get("run_id") != run_id
        ] + [record]
        self._save(state)
        return run_id

    def update(self, run_id: str, updates: dict) -> dict:
        """Merge updates into an existing record by run_id (create if missing)."""
        state = self._load()
        for record in state.setdefault("runs", []):
            if record.get("run_id") == run_id:
                record.update(updates)
                merged = record
                break
        else:
            merged = {"run_id": run_id, **updates}
            merged.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
            state["runs"].append(merged)
        self._save(state)
        return dict(merged)

    def all(self) -> list[dict]:
        """Return all records, newest first."""
        runs = self._load().get("runs", [])
        return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)

    def get(self, run_id: str) -> dict | None:
        """Return a record by run_id, or None."""
        for record in self._load().get("runs", []):
            if record.get("run_id") == run_id:
                return record
        return None

    def delete(self, run_id: str) -> dict:
        """Remove a record by run_id; returns how many rows were removed."""
        state = self._load()
        before = len(state.get("runs", []))
        state["runs"] = [r for r in state.get("runs", []) if r.get("run_id") != run_id]
        removed = before - len(state["runs"])
        if removed:
            self._save(state)
        return {"run_id": run_id, "removed": removed}

    @staticmethod
    def _new_id(kind: str) -> str:
        return f"{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def _load(self) -> dict:
        if not self.store_path.exists():
            return {"runs": []}
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"runs": []}

    def _save(self, state: dict) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.store_path)


_default_store: RunHistoryStore | None = None


def get_store() -> RunHistoryStore:
    """Return the process-wide default run history store."""
    global _default_store
    if _default_store is None:
        _default_store = RunHistoryStore()
    return _default_store
