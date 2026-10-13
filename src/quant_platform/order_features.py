"""Read-only order-flow feature extraction for normalized trades."""


def analyze_order_flow(trades: list[dict], large_trade_threshold: float = 0.0) -> dict[str, float | int]:
    if large_trade_threshold < 0:
        raise ValueError("large_trade_threshold must be non-negative")
    if not trades:
        return {"trade_count": 0, "buy_volume": 0.0, "sell_volume": 0.0, "volume_imbalance": 0.0, "vwap": 0.0, "large_trade_ratio": 0.0, "price_impact": 0.0}
    buy_volume = 0.0
    sell_volume = 0.0
    total_notional = 0.0
    large_volume = 0.0
    prices: list[float] = []
    for trade in trades:
        side = str(trade["side"]).lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"invalid trade side: {side}")
        price = float(trade["price"])
        quantity = float(trade["quantity"])
        if price <= 0 or quantity <= 0:
            raise ValueError("trade price and quantity must be positive")
        prices.append(price)
        total_notional += price * quantity
        if side == "buy":
            buy_volume += quantity
        else:
            sell_volume += quantity
        if quantity >= large_trade_threshold:
            large_volume += quantity
    total_volume = buy_volume + sell_volume
    return {
        "trade_count": len(trades),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "volume_imbalance": (buy_volume - sell_volume) / total_volume,
        "vwap": total_notional / total_volume,
        "large_trade_ratio": large_volume / total_volume,
        "price_impact": prices[-1] / prices[0] - 1,
    }
