from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    FUTURES = "futures"
    FX = "fx"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class MarketEvent:
    symbol: str
    asset_class: AssetClass
    price: float
    timestamp: datetime

    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: OrderSide
    quantity: float
    reason: str


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: OrderSide
    quantity: float
    strategy_id: str
    asset_class: AssetClass | None = None
    client_order_id: str | None = None
    reference_price: float | None = None

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.reference_price is not None and self.reference_price <= 0:
            raise ValueError("reference_price must be positive")

    @classmethod
    def from_signal(
        cls,
        signal: Signal,
        strategy_id: str,
        asset_class: AssetClass | None = None,
        reference_price: float | None = None,
    ) -> "OrderIntent":
        return cls(signal.symbol, signal.side, signal.quantity, strategy_id, asset_class, None, reference_price)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class OrderReceipt:
    symbol: str
    status: str
    gateway: str
    order_id: str | None = None
