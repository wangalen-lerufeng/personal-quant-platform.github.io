"""Versioned, serializable metadata for reproducible research objects."""

from dataclasses import asdict, dataclass, field
from typing import Any


def _validate_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    formula: str
    source: str
    timeframe: str
    parameters: dict[str, Any]
    uses_future_data: bool
    market: str
    version: int = 1

    def __post_init__(self) -> None:
        for value, field_name in ((self.name, "name"), (self.formula, "formula"), (self.source, "source"), (self.timeframe, "timeframe"), (self.market, "market")):
            _validate_text(value, field_name)
        if self.uses_future_data:
            raise ValueError("research features must not use future data")
        if self.version < 1:
            raise ValueError("version must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    definition: str
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    version: int = 1

    def __post_init__(self) -> None:
        _validate_text(self.name, "name")
        _validate_text(self.definition, "definition")
        if not self.feature_names:
            raise ValueError("factor must reference at least one feature")
        if self.version < 1:
            raise ValueError("version must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        return payload


@dataclass(frozen=True)
class StrategySpec:
    name: str
    strategy_type: str
    symbols: tuple[str, ...]
    timeframe: str
    entry_rules: tuple[str, ...]
    exit_rules: tuple[str, ...]
    risk_limits: dict[str, Any]
    version: int = 1

    def __post_init__(self) -> None:
        for value, field_name in ((self.name, "name"), (self.strategy_type, "strategy_type"), (self.timeframe, "timeframe")):
            _validate_text(value, field_name)
        if not self.symbols:
            raise ValueError("strategy must contain at least one symbol")
        if not self.entry_rules or not self.exit_rules:
            raise ValueError("strategy must define entry and exit rules")
        if self.version < 1:
            raise ValueError("version must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["entry_rules"] = list(self.entry_rules)
        payload["exit_rules"] = list(self.exit_rules)
        return payload
