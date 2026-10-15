"""Per-account holdings + PnL report for the paper simulation console.

Pure computation over an in-memory ``PaperSimulationSession``: no orders, no
network, no risk-engine mutation. Cost basis is reconstructed from fills and
positions are marked to the session's latest observed prices.
"""


def _cost_basis(fills) -> dict[str, float]:
    """Weighted-average entry price per symbol from buy/sell fills."""
    quantity: dict[str, float] = {}
    cost: dict[str, float] = {}
    for fill in fills:
        symbol = fill.symbol
        if fill.side == "buy":
            quantity[symbol] = quantity.get(symbol, 0.0) + fill.quantity
            cost[symbol] = cost.get(symbol, 0.0) + fill.quantity * fill.price
        else:
            if quantity.get(symbol, 0.0) > 1e-12:
                avg = cost.get(symbol, 0.0) / quantity[symbol]
                quantity[symbol] = max(0.0, quantity[symbol] - fill.quantity)
                cost[symbol] = max(0.0, cost[symbol] - fill.quantity * avg)
    return {symbol: cost[symbol] / quantity[symbol] for symbol in quantity if quantity[symbol] > 1e-12}


def build_account_report(session) -> dict:
    """Return holdings, cash, equity, and per-symbol PnL for a session."""
    positions = session.matcher.positions
    cash = float(session.matcher.cash)
    latest_prices = session.latest_prices
    basis = _cost_basis(session.matcher.fills)

    holdings = []
    total_market_value = 0.0
    total_cost_value = 0.0
    for symbol, quantity in positions.items():
        if quantity <= 0:
            continue
        cost = basis.get(symbol)
        latest = latest_prices.get(symbol)
        market_value = quantity * latest if latest is not None else None
        cost_value = quantity * cost if cost is not None else None
        pnl = None if market_value is None or cost_value is None else market_value - cost_value
        ret = None if latest is None or not cost else latest / cost - 1.0
        if market_value is not None:
            total_market_value += market_value
        if cost_value is not None:
            total_cost_value += cost_value
        holdings.append({
            "symbol": symbol,
            "quantity": quantity,
            "cost": cost,
            "latest": latest,
            "market_value": market_value,
            "pnl": pnl,
            "return": ret,
        })

    equity = cash + total_market_value
    total_pnl = total_market_value - total_cost_value
    return {
        "account_id": session.account_id,
        "cash": cash,
        "equity": equity,
        "market_value": total_market_value,
        "cost_value": total_cost_value,
        "unrealized_pnl": total_pnl,
        "unrealized_return": None if not total_cost_value else total_market_value / total_cost_value - 1.0,
        "holdings": sorted(holdings, key=lambda h: -(h["market_value"] or 0.0)),
        "has_pending_orders": bool(session.matcher._pending),
        "paper_only": True,
        "execution_enabled": False,
    }
