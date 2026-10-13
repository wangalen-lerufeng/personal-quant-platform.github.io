from uuid import uuid4

from ..domain import OrderIntent, OrderReceipt
from ..execution import OrderState


class _PaperGateway:
    name = "paper"

    def __init__(self):
        self._orders: dict[str, str] = {}

    def submit_order(self, intent: OrderIntent) -> OrderReceipt:
        order_id = str(uuid4())
        self._orders[order_id] = "accepted"
        return OrderReceipt(symbol=intent.symbol, status="accepted", gateway=self.name, order_id=order_id)

    def cancel_order(self, order_id: str):
        if order_id not in self._orders:
            raise KeyError(f"unknown order: {order_id}")
        self._orders[order_id] = "canceled"
        return OrderState.CANCELED

    def health_check(self) -> dict[str, object]:
        return {
            "gateway": self.name,
            "mode": "paper",
            "live_trading_enabled": False,
            "execution_enabled": False,
            "status": "ready",
        }


class QmtPaperGateway(_PaperGateway):
    name = "qmt-paper"


class CtpPaperGateway(_PaperGateway):
    name = "ctp-paper"


class Mt5PaperGateway(_PaperGateway):
    name = "mt5-paper"
