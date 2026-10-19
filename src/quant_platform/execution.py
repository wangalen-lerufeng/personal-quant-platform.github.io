from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import uuid4

from .domain import AssetClass, OrderIntent, OrderReceipt, OrderSide
from .execution_constraints import apply_lot_size


class OrderState(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


_TRANSITIONS = {
    OrderState.PENDING: {OrderState.ACCEPTED, OrderState.REJECTED, OrderState.CANCELED},
    OrderState.ACCEPTED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELED},
    OrderState.FILLED: set(),
    OrderState.CANCELED: set(),
    OrderState.REJECTED: set(),
}


@dataclass
class OrderLifecycle:
    order_id: str
    state: OrderState = OrderState.PENDING

    def transition(self, next_state: OrderState) -> None:
        if next_state not in _TRANSITIONS[self.state]:
            raise ValueError(f"invalid order transition: {self.state.value} -> {next_state.value}")
        self.state = next_state


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    order_id: str
    symbol: str


@dataclass(frozen=True)
class SimulatedFill:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float


@dataclass
class _PendingSimulationOrder:
    intent: OrderIntent
    remaining: float
    observed_events: int = 0


class PaperMatchingEngine:
    """Deterministic event-driven matcher for research and simulation only."""

    def __init__(
        self,
        initial_cash: float,
        latency_events: int = 0,
        max_participation_rate: float = 0.1,
        slippage_rate: float = 0.0,
        lot_size: float = 1.0,
    ):
        if initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")
        if latency_events < 0:
            raise ValueError("latency_events must be non-negative")
        if not 0 < max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1]")
        if not 0 <= slippage_rate < 1:
            raise ValueError("slippage_rate must be in [0, 1)")
        self.cash = float(initial_cash)
        self.latency_events = latency_events
        self.max_participation_rate = max_participation_rate
        self.slippage_rate = slippage_rate
        self.lot_size = float(lot_size)
        self.orders: dict[str, OrderLifecycle] = {}
        self._pending: dict[str, _PendingSimulationOrder] = {}
        self._intents: dict[str, OrderIntent] = {}
        self.positions: dict[str, float] = {}
        self.fills: list[SimulatedFill] = []

    def submit(self, intent: OrderIntent) -> str:
        order_id = str(uuid4())
        lifecycle = OrderLifecycle(order_id)
        lifecycle.transition(OrderState.ACCEPTED)
        self.orders[order_id] = lifecycle
        self._pending[order_id] = _PendingSimulationOrder(intent, intent.quantity)
        self._intents[order_id] = intent
        return order_id

    def intent_for(self, order_id: str) -> OrderIntent:
        try:
            return self._intents[order_id]
        except KeyError as error:
            raise KeyError(f"unknown order: {order_id}") from error

    def on_market(self, symbol: str, price: float, volume: float) -> list[SimulatedFill]:
        if price <= 0 or volume < 0:
            raise ValueError("market price must be positive and volume non-negative")
        fills: list[SimulatedFill] = []
        available_volume = volume * self.max_participation_rate
        for order_id, pending in list(self._pending.items()):
            if pending.intent.symbol != symbol or available_volume <= 0:
                continue
            if pending.observed_events < self.latency_events:
                pending.observed_events += 1
                continue
            side = pending.intent.side
            fill_price = price * (1 + self.slippage_rate if side.value == "buy" else 1 - self.slippage_rate)
            quantity = min(pending.remaining, available_volume)
            if side.value == "buy":
                quantity = min(quantity, self.cash / fill_price)
            else:
                quantity = min(quantity, max(self.positions.get(symbol, 0.0), 0.0))
            quantity = apply_lot_size(quantity, self.lot_size)
            if quantity <= 0:
                continue
            notional = quantity * fill_price
            if side.value == "buy":
                # 避免浮点误差使 notional 略高于 cash，导致现金出现极小负值
                notional = min(notional, self.cash)
            signed_quantity = quantity if side.value == "buy" else -quantity
            self.cash += -notional if side.value == "buy" else notional
            self.positions[symbol] = self.positions.get(symbol, 0.0) + signed_quantity
            pending.remaining -= quantity
            available_volume -= quantity
            next_state = OrderState.FILLED if pending.remaining <= 1e-12 else OrderState.PARTIALLY_FILLED
            self.orders[order_id].transition(next_state)
            fills.append(SimulatedFill(order_id, symbol, side.value, quantity, fill_price))
            if next_state is OrderState.FILLED:
                del self._pending[order_id]
        self.fills.extend(fills)
        return fills

    def snapshot(self) -> dict[str, object]:
        def serialize_intent(intent: OrderIntent) -> dict[str, object]:
            return {
                "symbol": intent.symbol,
                "side": intent.side.value,
                "quantity": intent.quantity,
                "strategy_id": intent.strategy_id,
                "asset_class": None if intent.asset_class is None else intent.asset_class.value,
                "client_order_id": intent.client_order_id,
                "reference_price": intent.reference_price,
            }

        return {
            "cash": self.cash,
            "latency_events": self.latency_events,
            "max_participation_rate": self.max_participation_rate,
            "slippage_rate": self.slippage_rate,
            "lot_size": self.lot_size,
            "positions": dict(self.positions),
            "orders": {
                order_id: lifecycle.state.value
                for order_id, lifecycle in self.orders.items()
            },
            "intents": {
                order_id: serialize_intent(intent)
                for order_id, intent in self._intents.items()
            },
            "pending": {
                order_id: {
                    "remaining": pending.remaining,
                    "observed_events": pending.observed_events,
                }
                for order_id, pending in self._pending.items()
            },
            "fills": [
                {
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "quantity": fill.quantity,
                    "price": fill.price,
                }
                for fill in self.fills
            ],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> "PaperMatchingEngine":
        engine = cls(
            initial_cash=float(snapshot["cash"]),
            latency_events=int(snapshot.get("latency_events", 0)),
            max_participation_rate=float(snapshot.get("max_participation_rate", 0.1)),
            slippage_rate=float(snapshot.get("slippage_rate", 0.0)),
            lot_size=float(snapshot.get("lot_size", 1.0)),
        )
        engine.cash = float(snapshot["cash"])
        engine.positions = {
            str(symbol): float(quantity)
            for symbol, quantity in dict(snapshot.get("positions", {})).items()
        }
        raw_intents = dict(snapshot.get("intents", {}))
        for order_id, raw in raw_intents.items():
            value = dict(raw)
            raw_asset_class = value.get("asset_class")
            engine._intents[str(order_id)] = OrderIntent(
                symbol=str(value["symbol"]),
                side=OrderSide(str(value["side"])),
                quantity=float(value["quantity"]),
                strategy_id=str(value["strategy_id"]),
                asset_class=(
                    None if raw_asset_class is None else AssetClass(str(raw_asset_class))
                ),
                client_order_id=value.get("client_order_id"),
                reference_price=(
                    None
                    if value.get("reference_price") is None
                    else float(value["reference_price"])
                ),
            )
        engine.orders = {
            str(order_id): OrderLifecycle(str(order_id), OrderState(str(state)))
            for order_id, state in dict(snapshot.get("orders", {})).items()
        }
        for order_id, raw in dict(snapshot.get("pending", {})).items():
            value = dict(raw)
            engine._pending[str(order_id)] = _PendingSimulationOrder(
                engine._intents[str(order_id)],
                float(value["remaining"]),
                int(value.get("observed_events", 0)),
            )
        engine.fills = [
            SimulatedFill(
                order_id=str(raw["order_id"]),
                symbol=str(raw["symbol"]),
                side=str(raw["side"]),
                quantity=float(raw["quantity"]),
                price=float(raw["price"]),
            )
            for raw in list(snapshot.get("fills", []))
        ]
        return engine


@runtime_checkable
class ExecutionGateway(Protocol):
    """Common contract implemented by paper and future broker gateways."""

    def submit_order(self, intent: OrderIntent) -> OrderReceipt: ...

    def cancel_order(self, order_id: str) -> OrderState: ...

    def get_positions(self) -> dict[str, float]: ...

    def health_check(self) -> dict[str, object]: ...


class InMemoryExecutionGateway:
    """Deterministic paper gateway used by tests and local development."""

    def __init__(self):
        self.audit_events: list[AuditEvent] = []
        self.orders: dict[str, OrderLifecycle] = {}
        self._symbols: dict[str, str] = {}
        self._intents: dict[str, OrderIntent] = {}
        self._positions: dict[str, float] = {}
        self._filled_quantities: dict[str, float] = {}

    def submit_order(self, intent: OrderIntent) -> OrderReceipt:
        order_id = str(uuid4())
        self.orders[order_id] = OrderLifecycle(order_id=order_id)
        self._symbols[order_id] = intent.symbol
        self._intents[order_id] = intent
        self.audit_events.append(AuditEvent("order_submitted", order_id, intent.symbol))
        return OrderReceipt(symbol=intent.symbol, status=OrderState.PENDING.value, gateway="paper", order_id=order_id)

    def cancel_order(self, order_id: str) -> OrderState:
        if order_id not in self.orders:
            raise KeyError(f"unknown order: {order_id}")
        lifecycle = self.orders[order_id]
        lifecycle.transition(OrderState.CANCELED)
        self.audit_events.append(AuditEvent("order_canceled", order_id, self._symbols[order_id]))
        return lifecycle.state

    def transition_order(self, order_id: str, next_state: OrderState) -> OrderState:
        if order_id not in self.orders:
            raise KeyError(f"unknown order: {order_id}")
        lifecycle = self.orders[order_id]
        lifecycle.transition(next_state)
        if next_state is OrderState.FILLED:
            intent = self._intents[order_id]
            remaining = intent.quantity - self._filled_quantities.get(order_id, 0.0)
            self._filled_quantities[order_id] = intent.quantity
            signed_quantity = remaining if intent.side.value == "buy" else -remaining
            self._positions[intent.symbol] = self._positions.get(intent.symbol, 0.0) + signed_quantity
        self.audit_events.append(
            AuditEvent(f"order_{next_state.value}", order_id, self._symbols[order_id])
        )
        return lifecycle.state

    def record_fill(self, order_id: str, quantity: float) -> OrderState:
        if order_id not in self.orders:
            raise KeyError(f"unknown order: {order_id}")
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        intent = self._intents[order_id]
        filled = self._filled_quantities.get(order_id, 0.0)
        if filled + quantity > intent.quantity:
            raise ValueError("fill quantity exceeds submitted order quantity")
        next_filled = filled + quantity
        next_state = OrderState.FILLED if next_filled == intent.quantity else OrderState.PARTIALLY_FILLED
        self.orders[order_id].transition(next_state)
        self._filled_quantities[order_id] = next_filled
        signed_quantity = quantity if intent.side.value == "buy" else -quantity
        self._positions[intent.symbol] = self._positions.get(intent.symbol, 0.0) + signed_quantity
        self.audit_events.append(AuditEvent(f"order_{next_state.value}", order_id, intent.symbol))
        return next_state

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def health_check(self) -> dict[str, object]:
        return {
            "gateway": "paper",
            "mode": "paper",
            "live_trading_enabled": False,
            "execution_enabled": False,
            "status": "ready",
        }
