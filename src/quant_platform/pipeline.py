"""Phase 9 end-to-end pipeline integration (step6).

Bridges the research pipeline's candidate pool + verification into target
positions, then routes every position change through the deterministic risk
engine before it can reach a paper simulation or an event backtest. Each stage
is recorded in an append-only audit trail.

Read-only by construction: the pipeline only produces risk-approved paper order
intents (``execution_enabled: false``) and never touches live trading.
"""

from dataclasses import dataclass
from typing import Any

from .domain import AssetClass, OrderIntent, OrderSide, Signal
from .position_sizing import clamp_adjustment
from .risk import RiskEngine


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    weight: float


@dataclass(frozen=True)
class AuditEntry:
    stage: str
    symbol: str | None
    detail: dict


@dataclass(frozen=True)
class ExecutionPlan:
    orders: tuple[OrderIntent, ...]
    rejected: tuple[dict, ...]
    audit: tuple[AuditEntry, ...]

    def to_dict(self) -> dict:
        return {
            "orders": [
                {"symbol": order.symbol, "side": order.side.value, "quantity": order.quantity}
                for order in self.orders
            ],
            "rejected": [dict(item) for item in self.rejected],
            "audit": [
                {"stage": entry.stage, "symbol": entry.symbol, "detail": dict(entry.detail)}
                for entry in self.audit
            ],
            "research_only": True,
            "execution_enabled": False,
        }


def build_targets(
    candidates,
    decisions,
    *,
    adjustment: float = 1.0,
) -> list[TargetPosition]:
    """Filter candidates to verified-``pass`` and assign equal-weight targets."""
    passed = [
        symbol
        for symbol in candidates
        if decisions.get(symbol) is not None and decisions[symbol].verdict == "pass"
    ]
    if not passed:
        return []
    weight = clamp_adjustment(adjustment) / len(passed)
    return [TargetPosition(symbol=symbol, weight=weight) for symbol in passed]


def plan_orders(
    current_positions,
    targets,
    risk_engine: RiskEngine,
    *,
    strategy_id: str = "phase9-pipeline",
    asset_class: AssetClass = AssetClass.ETF,
) -> ExecutionPlan:
    """Convert target positions into risk-gated order intents with an audit trail."""
    audit: list[AuditEntry] = [AuditEntry("plan_orders", None, {"target_count": len(targets)})]
    orders: list[OrderIntent] = []
    rejected: list[dict] = []
    for target in targets:
        current = float(current_positions.get(target.symbol, 0.0))
        delta = target.weight - current
        if abs(delta) < 1e-12:
            audit.append(AuditEntry("no_delta", target.symbol, {"delta": 0.0}))
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        intent = OrderIntent(
            symbol=target.symbol,
            side=side,
            quantity=abs(delta),
            strategy_id=strategy_id,
            asset_class=asset_class,
        )
        decision = risk_engine.evaluate(intent)
        audit.append(AuditEntry("risk_gate", target.symbol, {
            "side": side.value,
            "quantity": abs(delta),
            "allowed": decision.allowed,
            "reason": decision.reason,
        }))
        if decision.allowed:
            orders.append(intent)
        else:
            rejected.append({
                "symbol": target.symbol,
                "side": side.value,
                "quantity": abs(delta),
                "reason": decision.reason,
            })
    return ExecutionPlan(tuple(orders), tuple(rejected), tuple(audit))


def submit_to_simulation(plan: ExecutionPlan, session) -> dict:
    """Submit risk-approved intents to a paper simulation session."""
    submitted = []
    for intent in plan.orders:
        try:
            order_id = session.submit(intent)
            submitted.append({"symbol": intent.symbol, "order_id": order_id, "status": "accepted"})
        except RuntimeError as error:
            submitted.append({"symbol": intent.symbol, "status": "rejected", "reason": str(error)})
    return {
        "submitted": submitted,
        "order_count": len(submitted),
        "paper_only": True,
        "execution_enabled": False,
    }


class TargetWeightStrategy:
    """Emit BUY signals to reach scheduled target quantities; bridges to BacktestEngine."""

    def __init__(self, targets: dict[str, float]):
        self.targets = {
            symbol: float(quantity)
            for symbol, quantity in targets.items()
            if quantity > 0
        }
        self._entered: set[str] = set()

    def on_event(self, event) -> Signal | None:
        target = self.targets.get(event.symbol)
        if target is None or event.symbol in self._entered:
            return None
        self._entered.add(event.symbol)
        return Signal(event.symbol, OrderSide.BUY, target, "target_weight")

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._entered.clear()
        else:
            self._entered.discard(symbol)
