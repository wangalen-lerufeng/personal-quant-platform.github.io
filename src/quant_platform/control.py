from .config import RuntimeConfig
from .domain import OrderIntent, OrderSide
from .risk import RiskEngine
from .storage import SqliteAuditStore


class ControlPlane:
    def __init__(self, config: RuntimeConfig, risk_engine: RiskEngine, store: SqliteAuditStore, adapter_registry=None, experiment_store=None, position_book=None, strategy_registry=None, market_data_feed=None):
        self.config = config
        self.risk_engine = risk_engine
        self.store = store
        self.adapter_registry = adapter_registry
        self.experiment_store = experiment_store
        self.position_book = position_book
        self.strategy_registry = strategy_registry
        self.market_data_feed = market_data_feed

    def record_order(self, order_id: str, intent: OrderIntent, status: str) -> None:
        self.store.record_order(order_id, intent, status)

    def orders(self) -> list[dict]:
        return self.store.list_orders()

    def audit_events(self, order_id: str | None = None) -> list[dict]:
        return self.store.list_audit_events(order_id)

    def record_fill(self, symbol: str, side: OrderSide, quantity: float, price: float) -> None:
        if self.position_book is None:
            raise RuntimeError("position book is not configured")
        self.position_book.apply_fill(symbol, side, quantity, price)
        self.risk_engine.record_fill(symbol, side, quantity)

    def risk_state(self) -> dict[str, object]:
        limits = self.risk_engine.limits
        return {
            "kill_switch": limits.kill_switch,
            "max_order_quantity": limits.max_order_quantity,
            "max_order_value": limits.max_order_value,
            "max_position_quantity": limits.max_position_quantity,
            "max_daily_loss": limits.max_daily_loss,
            "max_drawdown": limits.max_drawdown,
            "daily_loss": self.risk_engine.daily_loss,
            "drawdown": self.risk_engine.drawdown,
            "positions": dict(self.risk_engine.positions),
        }

    def system_state(self) -> dict[str, object]:
        return {
            "environment": self.config.environment,
            "live_trading_enabled": self.config.live_trading_enabled,
            "ai_execution_tools_enabled": self.config.ai_execution_tools_enabled,
            "execution_enabled": self.config.live_trading_enabled or self.config.ai_execution_tools_enabled,
            "paper_only": not self.config.live_trading_enabled and not self.config.ai_execution_tools_enabled,
        }

    def adapter_health(self) -> list[dict[str, object]]:
        if self.adapter_registry is None:
            return []
        return self.adapter_registry.health()

    def experiments(self) -> list[dict]:
        if self.experiment_store is None:
            return []
        return self.experiment_store.list()

    def strategies(self) -> list[dict]:
        if self.strategy_registry is None:
            return []
        return self.strategy_registry.list()

    def portfolio_state(self, prices: dict[str, float] | None = None) -> dict[str, object]:
        if self.position_book is None:
            return {"positions": {}, "market_value": 0.0, "unrealized_pnl": 0.0, "equity": 0.0}
        if prices is None and self.market_data_feed is not None:
            prices = self.market_data_feed.latest_prices()
        return self.position_book.snapshot(prices or {}, require_prices=prices is not None)
