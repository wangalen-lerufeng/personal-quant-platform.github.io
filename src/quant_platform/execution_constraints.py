"""成交限制原语：整手、涨跌停、参与率、冲击成本、T+1。

这些纯函数描述 A 股/ETF 的可执行性约束，供回测与纸面成交复用。全部仅研究，
不产生真实订单。
"""


def apply_lot_size(quantity: float, lot_size: float = 100.0) -> float:
    """把数量向下取整到整手（A 股默认 100 股/手）。``lot_size <= 0`` 视为不限制。"""
    if lot_size is None or lot_size <= 0:
        return float(quantity)
    return float(int(quantity // lot_size) * lot_size)


def limit_price_band(previous_close: float, limit_pct: float) -> tuple[float, float]:
    """返回 (跌停价, 涨停价)。"""
    return previous_close * (1 - limit_pct), previous_close * (1 + limit_pct)


def is_limit_locked(side: str, price: float, previous_close: float, limit_pct: float) -> bool:
    """涨停无法买入、跌停无法卖出。"""
    lower, upper = limit_price_band(previous_close, limit_pct)
    tolerance = max(abs(previous_close), 1.0) * 1e-9
    if side in ("buy", "BUY") and price >= upper - tolerance:
        return True
    if side in ("sell", "SELL") and price <= lower + tolerance:
        return True
    return False


def participation_notional_cap(adv: float, participation_rate: float) -> float:
    """单笔订单成交额上限 = 日均成交额 × 参与率。"""
    return float(adv) * float(participation_rate)


def t_plus_1_sellable(available_quantity: float, bought_today_quantity: float) -> float:
    """T+1：当日买入部分当日不可卖。"""
    return max(0.0, float(available_quantity) - float(bought_today_quantity))
