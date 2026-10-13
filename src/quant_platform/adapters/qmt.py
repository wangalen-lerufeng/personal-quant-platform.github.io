from .base import MarketAdapter
from ..domain import AssetClass


class QmtAdapter(MarketAdapter):
    """Placeholder for QMT/PTrade A-share and ETF integration."""

    market = "mainland_equity"
    name = "qmt-paper"
    supported_asset_class = AssetClass.ETF
