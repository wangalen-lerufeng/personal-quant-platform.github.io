import tempfile
import unittest
from pathlib import Path

from quant_platform.run_history import RunHistoryStore


class RunHistoryStoreTests(unittest.TestCase):
    def test_add_and_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunHistoryStore(store_path=str(Path(tmp) / "history.json"))
            run_id = store.add({"kind": "phase9", "targets": [{"symbol": "A", "weight": 0.5}]})
            self.assertTrue(run_id.startswith("phase9-"))
            runs = store.all()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], run_id)
            self.assertIn("created_at", runs[0])

    def test_add_dedups_by_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunHistoryStore(store_path=str(Path(tmp) / "history.json"))
            store.add({"run_id": "fv-x", "kind": "forward", "sample_days": 1})
            store.add({"run_id": "fv-x", "kind": "forward", "sample_days": 3})
            runs = store.all()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["sample_days"], 3)

    def test_update_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunHistoryStore(store_path=str(Path(tmp) / "history.json"))
            store.add({"run_id": "fv-x", "kind": "forward", "metrics": {"a": 1}})
            store.update("fv-x", {"metrics": {"a": 2}, "sample_days": 5})
            runs = store.all()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["metrics"], {"a": 2})
            self.assertEqual(runs[0]["sample_days"], 5)

    def test_all_sorted_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunHistoryStore(store_path=str(Path(tmp) / "history.json"))
            store.add({"run_id": "p9-old", "kind": "phase9", "created_at": "2026-01-01T00:00:00"})
            store.add({"run_id": "p9-new", "kind": "phase9", "created_at": "2026-02-01T00:00:00"})
            runs = store.all()
            self.assertEqual(runs[0]["run_id"], "p9-new")
            self.assertEqual(runs[1]["run_id"], "p9-old")

    def test_delete_removes_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunHistoryStore(store_path=str(Path(tmp) / "history.json"))
            store.add({"run_id": "p9-a", "kind": "phase9"})
            store.add({"run_id": "fv-b", "kind": "forward"})
            result = store.delete("p9-a")
            self.assertEqual(result["removed"], 1)
            self.assertEqual([r["run_id"] for r in store.all()], ["fv-b"])

    def test_delete_missing_record_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunHistoryStore(store_path=str(Path(tmp) / "history.json"))
            store.add({"run_id": "p9-a", "kind": "phase9"})
            result = store.delete("does-not-exist")
            self.assertEqual(result["removed"], 0)
            self.assertEqual(len(store.all()), 1)


if __name__ == "__main__":
    unittest.main()
