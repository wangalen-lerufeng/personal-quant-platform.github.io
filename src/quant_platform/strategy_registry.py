from copy import deepcopy


class StrategyRegistry:
    """Small metadata registry for discoverable, versioned research strategies."""

    def __init__(self, strategies=None):
        self._strategies: dict[str, dict[str, object]] = {}
        for strategy_id, metadata in strategies or []:
            self.register(strategy_id, metadata)

    def register(self, strategy_id: str, metadata: dict[str, object] | None = None) -> None:
        if not strategy_id or strategy_id in self._strategies:
            raise ValueError("strategy_id must be non-empty and unique")
        self._strategies[strategy_id] = deepcopy(metadata or {})

    def list(self) -> list[dict[str, object]]:
        return deepcopy([
            {"strategy_id": strategy_id, **metadata}
            for strategy_id, metadata in self._strategies.items()
        ])
