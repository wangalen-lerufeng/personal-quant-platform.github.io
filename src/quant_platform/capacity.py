"""容量估计与冲击成本：最弱流动性标的、预期成交天数、冲击成本敏感性。

组合容量不能只用一个「Top 组成交额中位数 × 参与率」的单一数字，还需要报告
最弱流动性标的、按资产拆分的容量、目标下单金额下的预期成交天数，以及平方根
冲击成本对订单规模的敏感性。全部为研究估算，不等同于可成交资金规模。
"""

import math


def square_root_impact(notional: float, adv: float, impact_coefficient: float = 0.1) -> float:
    """平方根冲击模型：冲击成本比例 = coefficient * sqrt(notional / ADV)。

    ``adv`` 为该标的日均成交额，``notional`` 为订单名义金额。系数为研究启发值。
    """
    if notional is None or notional <= 0 or adv is None or adv <= 0:
        return 0.0
    return float(impact_coefficient * math.sqrt(float(notional) / float(adv)))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def estimate_capacity(
    daily_dollar_volumes: dict[str, list[float]],
    *,
    participation_rate: float = 0.01,
    target_notional: float | None = None,
    impact_coefficient: float = 0.1,
) -> dict:
    """基于每只资产的历史日成交额估计组合容量与冲击成本。

    ``daily_dollar_volumes`` 为 ``{symbol: [每日成交额]}``。返回每只资产的日均成交额、
    容量、预期成交天数与冲击成本，以及组合最弱流动性标的和总容量。
    """
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")

    per_asset: dict[str, dict] = {}
    for symbol, volumes in daily_dollar_volumes.items():
        values = [float(v) for v in volumes if v is not None and float(v) > 0]
        if not values:
            continue
        median_adv = _median(values)
        capacity = median_adv * participation_rate
        fill_days = (
            float(target_notional / capacity) if target_notional and capacity > 0 else None
        )
        impact = (
            square_root_impact(target_notional, median_adv, impact_coefficient)
            if target_notional
            else None
        )
        per_asset[str(symbol)] = {
            "median_daily_dollar_volume": round(median_adv, 2),
            "capacity": round(capacity, 2),
            "expected_fill_days": round(fill_days, 2) if fill_days is not None else None,
            "impact_cost": round(impact, 6) if impact is not None else None,
        }

    if not per_asset:
        return {"available": False, "participation_rate": participation_rate}

    weakest_symbol = min(per_asset, key=lambda s: per_asset[s]["median_daily_dollar_volume"])
    total_capacity = sum(item["capacity"] for item in per_asset.values())
    return {
        "available": True,
        "participation_rate": participation_rate,
        "impact_coefficient": impact_coefficient,
        "per_asset": per_asset,
        "weakest_asset": weakest_symbol,
        "weakest_median_daily_dollar_volume": per_asset[weakest_symbol]["median_daily_dollar_volume"],
        "total_capacity": round(total_capacity, 2),
    }
