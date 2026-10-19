"""交易成本模型：按资产类别拆分佣金、印花税、过户费、滑点和最低佣金。

所有费率为成交额的比例；``round_trip_rate`` 是买入+卖出两侧合计的比例，用于把
换手率折算为当日成本。研究用近似值，不代表实际可成交成本。
"""

_STOCK = {
    "asset_class": "stock",
    "label": "A 股",
    "buy_commission": 0.00025,   # 买入佣金 2.5bp
    "sell_commission": 0.00025,  # 卖出佣金 2.5bp
    "stamp_tax": 0.0005,         # 卖出印花税 5bp（仅卖出）
    "transfer_fee": 0.00001,     # 过户费 0.1bp/侧
    "slippage": 0.001,           # 滑点 10bp/侧
    "min_commission": 5.0,       # 最低佣金（元/笔）
}

_ETF = {
    "asset_class": "etf",
    "label": "ETF",
    "buy_commission": 0.0001,    # 买入佣金 1bp
    "sell_commission": 0.0001,   # 卖出佣金 1bp
    "stamp_tax": 0.0,            # ETF 免印花税
    "transfer_fee": 0.0,
    "slippage": 0.0005,          # 滑点 5bp/侧
    "min_commission": 5.0,
}

_HK = {
    "asset_class": "hk",
    "label": "港股",
    "buy_commission": 0.0005,
    "sell_commission": 0.0005,
    "stamp_tax": 0.001,          # 印花税约 0.1%（双边）用简化卖出侧
    "transfer_fee": 0.00002,
    "slippage": 0.001,
    "min_commission": 30.0,
}

_MODELS = {"stock": _STOCK, "etf": _ETF, "hk": _HK}


def build_cost_model(asset_class: str) -> dict:
    """Return the cost model for an asset class (falls back to stock)."""
    model = dict(_MODELS.get(asset_class, _STOCK))
    model["round_trip_rate"] = round(
        model["buy_commission"]
        + model["sell_commission"]
        + model["stamp_tax"]
        + 2 * model["transfer_fee"]
        + 2 * model["slippage"],
        8,
    )
    return model


def cost_breakdown(turnover_rate: float, model: dict) -> dict:
    """把一次换手率折算为各成本项（占组合比例）。"""
    t = float(turnover_rate)
    buy_commission = t * model["buy_commission"]
    sell_commission = t * model["sell_commission"]
    stamp_tax = t * model["stamp_tax"]
    transfer_fee = t * model["transfer_fee"] * 2
    slippage = t * model["slippage"] * 2
    total = buy_commission + sell_commission + stamp_tax + transfer_fee + slippage
    return {
        "buy_commission": round(buy_commission, 8),
        "sell_commission": round(sell_commission, 8),
        "stamp_tax": round(stamp_tax, 8),
        "transfer_fee": round(transfer_fee, 8),
        "slippage": round(slippage, 8),
        "total": round(total, 8),
    }
