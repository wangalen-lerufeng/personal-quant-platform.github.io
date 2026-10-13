from dataclasses import dataclass
from uuid import uuid4

from .domain import OrderIntent, OrderReceipt, RiskDecision
from .risk import RiskEngine
from .execution import ExecutionGateway, OrderState


@dataclass(frozen=True)
class PipelineResult:
    decision: RiskDecision
    receipt: OrderReceipt | None = None

    @property
    def allowed(self) -> bool:
        return self.decision.allowed


class TradingPipeline:
    """The only path from a strategy order intent to an execution gateway."""

    def __init__(self, risk_engine: RiskEngine, gateway: ExecutionGateway, audit_store=None, fill_handler=None):
        self.risk_engine = risk_engine
        self.gateway = gateway
        self.audit_store = audit_store
        self.fill_handler = fill_handler
        self._submitted_client_order_ids: set[str] = set()
        self._intents: dict[str, OrderIntent] = {}
        self._filled_quantities: dict[str, float] = {}

    def submit(self, intent: OrderIntent) -> PipelineResult:
        decision = self.risk_engine.evaluate(intent)
        if not decision.allowed:
            return PipelineResult(decision=decision)
        if intent.client_order_id is not None:
            persisted_duplicate = (
                self.audit_store is not None
                and self.audit_store.find_order_by_client_id(intent.client_order_id) is not None
            )
            if intent.client_order_id in self._submitted_client_order_ids or persisted_duplicate:
                return PipelineResult(decision=RiskDecision(False, "duplicate_order"))
        receipt = self.gateway.submit_order(intent)
        if receipt.status != "accepted":
            if self.audit_store is not None:
                audit_id = receipt.order_id or f"execution-rejected-{uuid4()}"
                self.audit_store.record_order(audit_id, intent, receipt.status)
            return PipelineResult(
                decision=RiskDecision(False, "execution_rejected"),
                receipt=receipt,
            )
        if self.audit_store is not None and receipt.order_id is not None:
            self.audit_store.record_order(receipt.order_id, intent, receipt.status)
        if intent.client_order_id is not None:
            self._submitted_client_order_ids.add(intent.client_order_id)
        if receipt.order_id is not None:
            self._intents[receipt.order_id] = intent
        return PipelineResult(decision=decision, receipt=receipt)

    def record_fill(self, order_id: str, price: float, quantity: float | None = None) -> None:
        """Forward a paper fill to the configured portfolio/control-plane sink."""
        if self.fill_handler is None:
            raise RuntimeError("fill handler is not configured")
        if order_id not in self._intents:
            raise KeyError(f"unknown order: {order_id}")
        intent = self._intents[order_id]
        fill_quantity = intent.quantity if quantity is None else quantity
        if fill_quantity <= 0:
            raise ValueError("fill quantity must be positive")
        filled = self._filled_quantities.get(order_id, 0.0)
        if filled + fill_quantity > intent.quantity:
            raise ValueError("fill quantity exceeds submitted order quantity")
        self.fill_handler(
            intent.symbol,
            intent.side,
            fill_quantity,
            price,
        )
        total_filled = filled + fill_quantity
        self._filled_quantities[order_id] = total_filled
        if self.audit_store is not None:
            status = OrderState.FILLED.value if total_filled == intent.quantity else OrderState.PARTIALLY_FILLED.value
            self.audit_store.update_status(order_id, status)

    def update_execution_status(self, order_id: str, status: str | OrderState) -> None:
        if self.audit_store is None:
            return
        normalized = status.value if isinstance(status, OrderState) else status
        if normalized not in {state.value for state in OrderState}:
            raise ValueError(f"unknown execution status: {normalized}")
        self.audit_store.update_status(order_id, normalized)
