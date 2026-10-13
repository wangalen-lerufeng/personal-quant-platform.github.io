from collections import defaultdict

from .domain import OrderSide


class PositionBook:
    def __init__(self):
        self._quantities: dict[str, float] = defaultdict(float)
        self._costs: dict[str, float] = defaultdict(float)
        self._realized_pnl = 0.0

    def apply_fill(self, symbol: str, side: OrderSide, quantity: float, price: float) -> None:
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")
        multiplier = 1 if side is OrderSide.BUY else -1
        next_quantity = self._quantities[symbol] + multiplier * quantity
        if next_quantity < 0:
            raise ValueError("cannot sell more than current position")
        if side is OrderSide.BUY:
            self._costs[symbol] += quantity * price
        elif next_quantity == 0:
            average_price = self._costs[symbol] / self._quantities[symbol]
            self._realized_pnl += (price - average_price) * quantity
            self._costs[symbol] = 0.0
        else:
            average_price = self._costs[symbol] / self._quantities[symbol]
            self._realized_pnl += (price - average_price) * quantity
            self._costs[symbol] = average_price * next_quantity
        self._quantities[symbol] = next_quantity

    def quantity(self, symbol: str) -> float:
        return self._quantities[symbol]

    def snapshot(self, prices: dict[str, float], require_prices: bool = True) -> dict[str, object]:
        positions: dict[str, dict[str, object]] = {}
        market_value = 0.0
        unrealized_pnl = 0.0
        for symbol, quantity in self._quantities.items():
            if quantity == 0:
                continue
            average_price = self._costs[symbol] / quantity
            if symbol not in prices:
                if require_prices:
                    raise ValueError(f"missing positive market price for {symbol}")
                positions[symbol] = {"quantity": quantity, "average_price": average_price}
                continue
            if prices[symbol] <= 0:
                raise ValueError(f"market price must be positive for {symbol}")
            current_price = prices[symbol]
            value = quantity * current_price
            pnl = value - self._costs[symbol]
            positions[symbol] = {
                "quantity": quantity,
                "average_price": average_price,
                "market_price": current_price,
                "market_value": value,
                "unrealized_pnl": pnl,
            }
            market_value += value
            unrealized_pnl += pnl
        has_prices = all("market_price" in position for position in positions.values())
        return {
            "positions": positions,
            "market_value": market_value if has_prices else None,
            "unrealized_pnl": unrealized_pnl if has_prices else None,
            "realized_pnl": self._realized_pnl,
            "total_pnl": (self._realized_pnl + unrealized_pnl) if has_prices else None,
            "equity": market_value if has_prices else None,
        }
