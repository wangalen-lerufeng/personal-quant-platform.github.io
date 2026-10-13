from copy import deepcopy
import json
import sqlite3


class ExperimentStore:
    def __init__(self):
        self._results: list[dict] = []

    def save(self, strategy_id: str, metrics: dict) -> None:
        self._results.append({"strategy_id": strategy_id, "metrics": deepcopy(metrics)})

    def list(self) -> list[dict]:
        return deepcopy(self._results)


class SqliteExperimentStore:
    def __init__(self, database: str):
        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS experiments (strategy_id TEXT NOT NULL, metrics_json TEXT NOT NULL)"
        )
        self.connection.commit()

    def save(self, strategy_id: str, metrics: dict) -> None:
        self.connection.execute(
            "INSERT INTO experiments(strategy_id, metrics_json) VALUES (?, ?)",
            (strategy_id, json.dumps(metrics, sort_keys=True)),
        )
        self.connection.commit()

    def list(self) -> list[dict]:
        rows = self.connection.execute("SELECT strategy_id, metrics_json FROM experiments ORDER BY rowid").fetchall()
        return [{"strategy_id": row["strategy_id"], "metrics": json.loads(row["metrics_json"])} for row in rows]
