from dataclasses import dataclass

from .domain import AssetClass, OrderIntent


class RuleViolation(ValueError):
    pass


@dataclass(frozen=True)
class MarketRule:
    minimum_quantity: float
    quantity_step: float


class MarketRuleBook:
    def __init__(self):
        self._rules = {
            AssetClass.ETF: MarketRule(minimum_quantity=100, quantity_step=100),
            AssetClass.FUTURES: MarketRule(minimum_quantity=1, quantity_step=1),
            AssetClass.FX: MarketRule(minimum_quantity=0.01, quantity_step=0.01),
        }

    def validate(self, asset_class: AssetClass, intent: OrderIntent) -> None:
        rule = self._rules[asset_class]
        if intent.quantity < rule.minimum_quantity:
            raise RuleViolation(f"quantity below minimum for {asset_class.value}")
        units = intent.quantity / rule.quantity_step
        if abs(units - round(units)) > 1e-9:
            raise RuleViolation(f"quantity must use step {rule.quantity_step} for {asset_class.value}")
