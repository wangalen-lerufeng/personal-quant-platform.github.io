from dataclasses import dataclass
from typing import Any, Mapping
from pathlib import Path


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field} must be a boolean")


@dataclass(frozen=True)
class RuntimeConfig:
    environment: str = "development"
    live_trading_enabled: bool = False
    ai_execution_tools_enabled: bool = False
    market_data_enabled: bool = False
    market_data_url: str = ""
    market_data_api_key_env: str = "MARKET_DATA_API_KEY"
    market_data_timeout_seconds: float = 10.0
    ai_model_enabled: bool = False
    ai_provider: str = "openai-compatible"
    ai_base_url: str = ""
    ai_model: str = ""
    ai_api_key_env: str = "AI_API_KEY"
    ai_temperature: float = 0.2
    ai_max_tokens: int = 2000
    ai_timeout_seconds: float = 90.0
    database_path: str = ":memory:"

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required to load a config file") from error
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, Mapping):
            raise ValueError("config file root must be a mapping")
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RuntimeConfig":
        app = values.get("app", {})
        ai = values.get("ai", {})
        data = values.get("market_data", {})
        model = values.get("model", values.get("ai_model", {}))
        storage = values.get("storage", {})
        if not isinstance(app, Mapping):
            raise ValueError("app configuration must be a mapping")
        if not isinstance(ai, Mapping):
            raise ValueError("ai configuration must be a mapping")
        if not isinstance(data, Mapping):
            raise ValueError("market_data configuration must be a mapping")
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        if not isinstance(storage, Mapping):
            raise ValueError("storage configuration must be a mapping")
        environment = values.get("environment", app.get("environment", "development"))
        live_enabled = values.get("live_trading_enabled", app.get("live_trading_enabled", False))
        ai_enabled = values.get(
            "ai_execution_tools_enabled",
            ai.get("execution_tools_enabled", False),
        )
        data_enabled = values.get("market_data_enabled", data.get("enabled", False))
        data_url = values.get("market_data_url", data.get("url", ""))
        data_key_env = values.get("market_data_api_key_env", data.get("api_key_env", "MARKET_DATA_API_KEY"))
        data_timeout = values.get("market_data_timeout_seconds", data.get("timeout_seconds", 10.0))
        model_enabled = values.get("ai_model_enabled", model.get("enabled", False))
        return cls(
            environment=str(environment),
            live_trading_enabled=_as_bool(live_enabled, "live_trading_enabled"),
            ai_execution_tools_enabled=_as_bool(ai_enabled, "ai_execution_tools_enabled"),
            market_data_enabled=_as_bool(data_enabled, "market_data.enabled"),
            market_data_url=str(data_url),
            market_data_api_key_env=str(data_key_env),
            market_data_timeout_seconds=float(data_timeout),
            ai_model_enabled=_as_bool(model_enabled, "model.enabled"),
            ai_provider=str(model.get("provider", "openai-compatible")),
            ai_base_url=str(model.get("base_url", "")),
            ai_model=str(model.get("name", model.get("model", ""))),
            ai_api_key_env=str(model.get("api_key_env", "AI_API_KEY")),
            ai_temperature=float(model.get("temperature", 0.2)),
            ai_max_tokens=int(model.get("max_tokens", 2000)),
            ai_timeout_seconds=float(model.get("timeout_seconds", 90.0)),
            database_path=str(storage.get("database", ":memory:")),
        )
