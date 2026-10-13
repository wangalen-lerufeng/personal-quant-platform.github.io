import unittest
from pathlib import Path

from quant_platform.config import RuntimeConfig


class ConfigTests(unittest.TestCase):
    def test_example_configuration_allows_deep_research_output(self):
        config = RuntimeConfig.from_file(Path("configs/settings.example.yaml"))
        self.assertGreaterEqual(config.ai_max_tokens, 6000)
        self.assertLessEqual(config.ai_max_tokens, 8000)

    def test_defaults_disable_live_trading_and_agent_execution(self):
        config = RuntimeConfig()

        self.assertFalse(config.live_trading_enabled)
        self.assertFalse(config.ai_execution_tools_enabled)
        self.assertFalse(config.market_data_enabled)

    def test_market_data_configuration_is_read_from_nested_mapping(self):
        config = RuntimeConfig.from_mapping({"market_data": {"enabled": True, "url": "https://quotes.test", "timeout_seconds": 3}})
        self.assertTrue(config.market_data_enabled)
        self.assertEqual(config.market_data_url, "https://quotes.test")
        self.assertEqual(config.market_data_timeout_seconds, 3)

    def test_model_configuration_is_read_without_storing_secret(self):
        config = RuntimeConfig.from_mapping({"model": {"enabled": True, "provider": "deepseek", "base_url": "https://api.test/v1", "name": "deepseek-chat", "temperature": 0.1}})
        self.assertTrue(config.ai_model_enabled)
        self.assertEqual(config.ai_model, "deepseek-chat")
        self.assertEqual(config.ai_temperature, 0.1)
        self.assertEqual(config.environment, "development")

    def test_storage_database_path_is_configurable(self):
        config = RuntimeConfig.from_mapping({"storage": {"database": "data/platform.db"}})

        self.assertEqual(config.database_path, "data/platform.db")

    def test_live_trading_requires_explicit_enablement(self):
        config = RuntimeConfig.from_mapping({"environment": "production", "live_trading_enabled": True})

        self.assertTrue(config.live_trading_enabled)
        self.assertFalse(config.ai_execution_tools_enabled)

    def test_string_false_does_not_enable_safety_sensitive_flags(self):
        config = RuntimeConfig.from_mapping(
            {
                "live_trading_enabled": "false",
                "ai_execution_tools_enabled": "false",
            }
        )

        self.assertFalse(config.live_trading_enabled)
        self.assertFalse(config.ai_execution_tools_enabled)

    def test_unknown_string_boolean_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeConfig.from_mapping({"live_trading_enabled": "maybe"})

    def test_nested_example_configuration_is_supported(self):
        config = RuntimeConfig.from_mapping(
            {
                "app": {"environment": "development", "live_trading_enabled": False},
                "ai": {"execution_tools_enabled": False},
            }
        )

        self.assertEqual(config.environment, "development")
        self.assertFalse(config.live_trading_enabled)
        self.assertFalse(config.ai_execution_tools_enabled)

    def test_invalid_nested_sections_are_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeConfig.from_mapping({"app": "not-a-mapping"})
