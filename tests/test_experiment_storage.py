import unittest

from quant_platform.experiments import SqliteExperimentStore


class ExperimentStorageTests(unittest.TestCase):
    def test_sqlite_experiment_store_persists_metrics(self):
        store = SqliteExperimentStore(":memory:")
        store.save("trend-v1", {"trade_count": 2, "win_rate": 0.5})

        results = store.list()

        self.assertEqual(results[0]["strategy_id"], "trend-v1")
        self.assertEqual(results[0]["metrics"]["trade_count"], 2)
