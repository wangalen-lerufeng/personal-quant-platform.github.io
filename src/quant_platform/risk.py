from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .domain import OrderIntent, OrderSide, RiskDecision
from .config import _as_bool
from .market_rules import RuleViolation


@dataclass(frozen=True)
class RiskLimits:
    kill_switch: bool = False
    max_order_quantity: float = 1_000_000
    max_position_quantity: float | None = None
    max_order_value: float | None = None
    max_daily_loss: float | None = None
    max_drawdown: float | None = None

    def __post_init__(self):
        if self.max_order_quantity <= 0:
            raise ValueError("max_order_quantity must be positive")
        for name in ("max_order_value", "max_daily_loss", "max_drawdown"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_position_quantity is not None and self.max_position_quantity <= 0:
            raise ValueError("max_position_quantity must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RiskLimits":
        nested = values.get("risk", values)
        if not isinstance(nested, Mapping):
            raise ValueError("risk configuration must be a mapping")

        def number(name: str, default):
            value = nested.get(name, default)
            return None if value is None else float(value)

        return cls(
            kill_switch=_as_bool(nested.get("kill_switch", False), "kill_switch"),
            max_order_quantity=number("max_order_quantity", 1_000_000),
            max_position_quantity=number("max_position_quantity", None),
            max_order_value=number("max_order_value", None),
            max_daily_loss=number("max_daily_loss", None),
            max_drawdown=number("max_drawdown", None),
        )


class RiskEngine:
    def __init__(self, limits: RiskLimits, market_rules=None):
        self.limits = limits
        self.market_rules = market_rules
        self.daily_loss = 0.0
        self.drawdown = 0.0
        self.positions: dict[str, float] = {}

    def record_fill(self, symbol: str, side: OrderSide, quantity: float) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        signed = quantity if side is OrderSide.BUY else -quantity
        self.positions[symbol] = self.positions.get(symbol, 0.0) + signed

    def record_pnl(self, pnl: float) -> None:
        if pnl < 0:
            self.daily_loss += -pnl

    def record_drawdown(self, drawdown: float) -> None:
        self.drawdown = max(self.drawdown, drawdown)

    def start_new_day(self) -> None:
        self.daily_loss = 0.0

    def evaluate(self, intent: OrderIntent) -> RiskDecision:
        if self.limits.kill_switch:
            return RiskDecision(False, "kill_switch_active")
        if self.limits.max_daily_loss is not None and self.daily_loss >= self.limits.max_daily_loss:
            return RiskDecision(False, "daily_loss_limit")
        if self.limits.max_drawdown is not None and self.drawdown >= self.limits.max_drawdown:
            return RiskDecision(False, "drawdown_limit")
        if intent.quantity > self.limits.max_order_quantity:
            return RiskDecision(False, "order_quantity_exceeded")
        if self.limits.max_position_quantity is not None:
            signed = intent.quantity if intent.side is OrderSide.BUY else -intent.quantity
            resulting = self.positions.get(intent.symbol, 0.0) + signed
            if abs(resulting) > self.limits.max_position_quantity:
                return RiskDecision(False, "position_quantity_exceeded")
        if self.limits.max_order_value is not None:
            if intent.reference_price is None:
                return RiskDecision(False, "reference_price_required")
            if intent.quantity * intent.reference_price > self.limits.max_order_value:
                return RiskDecision(False, "order_value_exceeded")
        if self.market_rules is not None and intent.asset_class is not None:
            try:
                self.market_rules.validate(intent.asset_class, intent)
            except RuleViolation:
                return RiskDecision(False, "market_rule_violation")
        return RiskDecision(True, "approved")


class RiskMonitor:
    """Tracks account-level paper risk without granting execution authority."""

    def __init__(self, limits: RiskLimits, initial_equity: float):
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        self.limits = limits
        self.day_start_equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.current_equity = float(initial_equity)

    def observe(self, equity: float) -> dict[str, object]:
        if equity < 0:
            raise ValueError("equity must be non-negative")
        self.current_equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        daily_loss = max(self.day_start_equity - self.current_equity, 0.0)
        drawdown = max(self.peak_equity - self.current_equity, 0.0)
        alerts: list[str] = []
        if self.limits.kill_switch:
            alerts.append("kill_switch_active")
        if self.limits.max_daily_loss is not None and daily_loss >= self.limits.max_daily_loss:
            alerts.append("daily_loss_limit")
        if self.limits.max_drawdown is not None and drawdown >= self.limits.max_drawdown:
            alerts.append("drawdown_limit")
        if alerts:
            risk_level = "halt"
        elif drawdown > 0 or daily_loss > 0:
            risk_level = "elevated"
        else:
            risk_level = "normal"
        return {
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "daily_loss": daily_loss,
            "drawdown": drawdown,
            "risk_level": risk_level,
            "alerts": alerts,
            "execution_enabled": False,
        }

    def start_new_day(self) -> None:
        self.day_start_equity = self.current_equity


class OperationalRiskGuard:
    """Tracks operational paper-trading hazards without execution authority."""

    def __init__(
        self,
        max_consecutive_losses: int | None = None,
        market_data_timeout_seconds: float | None = None,
        max_api_failures: int | None = None,
    ):
        if max_consecutive_losses is not None and max_consecutive_losses <= 0:
            raise ValueError("max_consecutive_losses must be positive")
        if market_data_timeout_seconds is not None and market_data_timeout_seconds <= 0:
            raise ValueError("market_data_timeout_seconds must be positive")
        if max_api_failures is not None and max_api_failures <= 0:
            raise ValueError("max_api_failures must be positive")
        self.max_consecutive_losses = max_consecutive_losses
        self.market_data_timeout_seconds = market_data_timeout_seconds
        self.max_api_failures = max_api_failures
        self.consecutive_losses = 0
        self.api_failures = 0
        self.last_market_data_at: datetime | None = None
        self._manual_alert: str | None = None

    def record_trade_pnl(self, pnl: float) -> None:
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0

    def record_market_event(self, timestamp: datetime | None = None) -> None:
        observed_at = timestamp or datetime.now(timezone.utc)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("market event timestamp must be timezone-aware")
        self.last_market_data_at = observed_at

    def record_api_result(self, success: bool) -> None:
        self.api_failures = 0 if success else self.api_failures + 1

    def activate_kill_switch(self, reason: str = "manual_kill_switch") -> None:
        if not reason.strip():
            raise ValueError("kill switch reason is required")
        self._manual_alert = reason

    def reset_kill_switch(self) -> None:
        self._manual_alert = None

    def status(self, now: datetime | None = None) -> dict[str, object]:
        checked_at = now or datetime.now(timezone.utc)
        if checked_at.tzinfo is None or checked_at.utcoffset() is None:
            raise ValueError("status timestamp must be timezone-aware")
        alerts: list[str] = []
        if self._manual_alert is not None:
            alerts.append(self._manual_alert)
        if (
            self.max_consecutive_losses is not None
            and self.consecutive_losses >= self.max_consecutive_losses
        ):
            alerts.append("consecutive_loss_limit")
        if (
            self.market_data_timeout_seconds is not None
            and self.last_market_data_at is not None
            and (checked_at - self.last_market_data_at).total_seconds()
            > self.market_data_timeout_seconds
        ):
            alerts.append("market_data_stale")
        if self.max_api_failures is not None and self.api_failures >= self.max_api_failures:
            alerts.append("api_failure_limit")
        return {
            "halted": bool(alerts),
            "risk_level": "halt" if alerts else "normal",
            "alerts": alerts,
            "consecutive_losses": self.consecutive_losses,
            "api_failures": self.api_failures,
            "last_market_data_at": (
                None if self.last_market_data_at is None else self.last_market_data_at.isoformat()
            ),
            "execution_enabled": False,
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "max_consecutive_losses": self.max_consecutive_losses,
            "market_data_timeout_seconds": self.market_data_timeout_seconds,
            "max_api_failures": self.max_api_failures,
            "consecutive_losses": self.consecutive_losses,
            "api_failures": self.api_failures,
            "last_market_data_at": (
                None if self.last_market_data_at is None else self.last_market_data_at.isoformat()
            ),
            "manual_alert": self._manual_alert,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "OperationalRiskGuard":
        guard = cls(
            max_consecutive_losses=snapshot.get("max_consecutive_losses"),
            market_data_timeout_seconds=snapshot.get("market_data_timeout_seconds"),
            max_api_failures=snapshot.get("max_api_failures"),
        )
        guard.consecutive_losses = int(snapshot.get("consecutive_losses", 0))
        guard.api_failures = int(snapshot.get("api_failures", 0))
        raw_timestamp = snapshot.get("last_market_data_at")
        guard.last_market_data_at = (
            None if raw_timestamp is None else datetime.fromisoformat(str(raw_timestamp))
        )
        raw_alert = snapshot.get("manual_alert")
        guard._manual_alert = None if raw_alert is None else str(raw_alert)
        return guard
