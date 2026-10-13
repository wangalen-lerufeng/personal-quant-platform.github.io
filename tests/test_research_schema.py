import unittest

from quant_platform.research_schema import FeatureDefinition, FactorDefinition, StrategySpec


class ResearchSchemaTests(unittest.TestCase):
    def test_feature_requires_non_future_looking_definition(self):
        feature = FeatureDefinition(
            name="momentum_20d",
            formula="close / close.shift(20) - 1",
            source="ohlcv",
            timeframe="1d",
            parameters={"window": 20},
            uses_future_data=False,
            market="etf",
        )
        self.assertEqual(feature.version, 1)
        self.assertFalse(feature.uses_future_data)

    def test_future_looking_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            FeatureDefinition(
                name="bad",
                formula="close.shift(-1)",
                source="ohlcv",
                timeframe="1d",
                parameters={},
                uses_future_data=True,
                market="etf",
            )

    def test_factor_and_strategy_are_serializable(self):
        factor = FactorDefinition(name="value", definition="pe_inverse", feature_names=("pe",))
        strategy = StrategySpec(
            name="single_asset_cta",
            strategy_type="trend_following",
            symbols=("510300.SH",),
            timeframe="1d",
            entry_rules=("close > high.shift(55)",),
            exit_rules=("close < low.shift(20)",),
            risk_limits={"risk_per_trade": 0.005},
        )
        self.assertEqual(factor.to_dict()["name"], "value")
        self.assertEqual(strategy.to_dict()["symbols"], ["510300.SH"])

