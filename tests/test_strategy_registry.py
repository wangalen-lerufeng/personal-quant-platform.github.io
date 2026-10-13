import unittest

from quant_platform.strategy_registry import StrategyRegistry


class StrategyRegistryTests(unittest.TestCase):
    def test_registry_lists_metadata_without_execution_capability(self):
        registry = StrategyRegistry()
        registry.register("momentum", {"status": "research-only", "execution_enabled": False})

        self.assertEqual(registry.list(), [{"strategy_id": "momentum", "status": "research-only", "execution_enabled": False}])

    def test_registry_rejects_duplicate_ids(self):
        registry = StrategyRegistry()
        registry.register("momentum")

        with self.assertRaises(ValueError):
            registry.register("momentum")

    def test_registry_list_returns_isolated_metadata(self):
        registry = StrategyRegistry()
        registry.register("momentum", {"tags": ["trend"]})

        listed = registry.list()
        listed[0]["tags"].append("mutated")

        self.assertEqual(registry.list()[0]["tags"], ["trend"])
