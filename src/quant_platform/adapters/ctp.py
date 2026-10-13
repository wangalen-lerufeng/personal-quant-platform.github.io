from .base import MarketAdapter
from ..domain import AssetClass


class CtpAdapter(MarketAdapter):
    """Placeholder for domestic futures CTP integration."""

    market = "mainland_futures"
    name = "ctp-paper"
    supported_asset_class = AssetClass.FUTURES
