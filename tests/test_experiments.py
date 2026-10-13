import unittest

from quant_platform.experiments import ExperimentStore


class ExperimentTests(unittest.TestCase):
    def test_store_saves_and_lists_experiment_result(self):
        store = ExperimentStore()
        store.save("trend-v1", {"trade_count": 3, "total_return": 0.12})

        results = store.list()

        self.assertEqual(results[0]["strategy_id"], "trend-v1")
        self.assertEqual(results[0]["metrics"]["trade_count"], 3)
