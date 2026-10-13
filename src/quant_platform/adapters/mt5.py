from .base import MarketAdapter
from ..domain import AssetClass


class Mt5Adapter(MarketAdapter):
    """Placeholder for MT5 broker-symbol integration."""

    market = "mt5_broker_symbols"
    name = "mt5-paper"
    supported_asset_class = AssetClass.FX
