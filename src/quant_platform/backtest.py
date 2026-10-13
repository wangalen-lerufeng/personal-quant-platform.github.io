from dataclasses import dataclass, field
from datetime import datetime

from .data_quality import validate_events
from .domain import AssetClass, MarketEvent, OrderSide, Signal
from .market_calendar import MarketCalendar


def chronological_split(events: list[MarketEvent], in_sample_ratio: float = 0.7) -> tuple[list[MarketEvent], list[MarketEvent]]:
    """Split ordered events without shuffling, preserving temporal causality."""
    if not 0 < in_sample_ratio < 1:
        raise ValueError("in_sample_ratio must be between 0 and 1")
    validate_events(events)
    split_index = int(len(events) * in_sample_ratio)
    if split_index <= 0 or split_index >= len(events):
        raise ValueError("both in-sample and out-of-sample sets must be non-empty")
    return list(events[:split_index]), list(events[split_index:])


@dataclass(frozen=True)
class FuturesRolloverRule:
    expiring_symbol: str
    successor_symbol: str
    effective_at: datetime

    def __post_init__(self):
        if self.effective_at.tzinfo is None or self.effective_at.utcoffset() is None:
            raise ValueError("rollover effective_at must be timezone-aware")
        if not self.expiring_symbol or not self.successor_symbol:
            raise ValueError("rollover symbols are required")


class MarketExecutionPolicy:
    """Conservative exchange constraints for event-driven research backtests."""

    _MARKETS = {
        AssetClass.ETF: "mainland_equity",
        AssetClass.FUTURES: "mainland_futures",
        AssetClass.FX: "mt5_broker_symbols",
    }

    def __init__(
        self,
        calendar: MarketCalendar | None = None,
        previous_closes: dict[str, float] | None = None,
        price_limit_percentages: dict[str, float] | None = None,
        rollover_rules: list[FuturesRolloverRule] | None = None,
        auto_rollover: bool = False,
    ):
        self.calendar = calendar or MarketCalendar()
        self.previous_closes = dict(previous_closes or {})
        self.price_limit_percentages = dict(price_limit_percentages or {})
        self.rollover_rules = list(rollover_rules or [])
        self.auto_rollover = bool(auto_rollover)
        for symbol, price in self.previous_closes.items():
            if price <= 0:
                raise ValueError(f"previous close must be positive for {symbol}")
        for symbol, percentage in self.price_limit_percentages.items():
            if not 0 < percentage < 1:
                raise ValueError(f"price limit percentage must be in (0, 1) for {symbol}")

    def validate_event(self, event: MarketEvent) -> tuple[bool, str | None]:
        market = self._MARKETS[event.asset_class]
        if not self.calendar.is_open(market, event.timestamp):
            return False, "market_closed"
        for rule in self.rollover_rules:
            if event.symbol == rule.expiring_symbol and event.timestamp >= rule.effective_at:
                return False, f"expired_contract_use_{rule.successor_symbol}"
        return True, None

    def validate_signal(self, event: MarketEvent, signal: Signal) -> tuple[bool, str | None]:
        previous_close = self.previous_closes.get(event.symbol)
        limit = self.price_limit_percentages.get(event.symbol)
        if previous_close is None or limit is None:
            return True, None
        upper = previous_close * (1 + limit)
        lower = previous_close * (1 - limit)
        tolerance = max(abs(previous_close), 1.0) * 1e-9
        if signal.side is OrderSide.BUY and event.price >= upper - tolerance:
            return False, "buy_blocked_at_upper_price_limit"
        if signal.side is OrderSide.SELL and event.price <= lower + tolerance:
            return False, "sell_blocked_at_lower_price_limit"
        return True, None

    def rollover_for_successor(self, event: MarketEvent) -> FuturesRolloverRule | None:
        if not self.auto_rollover or event.asset_class is not AssetClass.FUTURES:
            return None
        return next((
            rule for rule in self.rollover_rules
            if event.symbol == rule.successor_symbol and event.timestamp >= rule.effective_at
        ), None)


@dataclass(frozen=True)
class BacktestResult:
    signal_count: int
    trade_count: int
    total_return: float
    trades: list[dict]
    win_rate: float
    max_drawdown: float
    equity_curve: list[float]
    restrictions: list[dict] = field(default_factory=list)
    rollovers: list[dict] = field(default_factory=list)

    @property
    def restriction_count(self) -> int:
        return len(self.restrictions)

    @property
    def rollover_count(self) -> int:
        return len(self.rollovers)

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-friendly research result representation."""
        gross_profit = sum(trade["return"] for trade in self.trades if trade["return"] > 0)
        gross_loss = sum(-trade["return"] for trade in self.trades if trade["return"] < 0)
        profit_factor = None if gross_loss == 0 else gross_profit / gross_loss
        loss_streak = 0
        max_loss_streak = 0
        for trade in self.trades:
            loss_streak = loss_streak + 1 if trade["return"] < 0 else 0
            max_loss_streak = max(max_loss_streak, loss_streak)
        return {
            "signal_count": self.signal_count,
            "trade_count": self.trade_count,
            "total_return": self.total_return,
            "trades": [dict(trade) for trade in self.trades],
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "profit_factor": profit_factor,
            "average_trade_return": self.total_return / self.trade_count if self.trade_count else 0.0,
            "total_commission": sum(trade.get("commission_cost", 0.0) for trade in self.trades),
            "total_slippage": sum(trade.get("slippage_cost", 0.0) for trade in self.trades),
            "max_consecutive_losses": max_loss_streak,
            "equity_curve": list(self.equity_curve),
            "restriction_count": self.restriction_count,
            "restrictions": [dict(item) for item in self.restrictions],
            "rollover_count": self.rollover_count,
            "rollovers": [dict(item) for item in self.rollovers],
            "research_only": True,
            "execution_enabled": False,
        }


class BacktestEngine:
    def __init__(
        self,
        strategy,
        commission_rate: float = 0.0,
        slippage_rate: float = 0.0,
        execution_policy: MarketExecutionPolicy | None = None,
    ):
        if commission_rate < 0 or commission_rate >= 1:
            raise ValueError("commission_rate must be in [0, 1)")
        if slippage_rate < 0 or slippage_rate >= 1:
            raise ValueError("slippage_rate must be in [0, 1)")
        self.strategy = strategy
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.execution_policy = execution_policy

    def run(self, events: list[MarketEvent]) -> BacktestResult:
        validate_events(events)
        reset = getattr(self.strategy, "reset", None)
        if callable(reset):
            reset()
        signal_count = 0
        entries: dict[str, tuple[float, float, float, int]] = {}
        last_prices: dict[str, float] = {}
        trades: list[dict] = []
        equity_curve: list[float] = []
        restrictions: list[dict] = []
        rollovers: list[dict] = []
        realized_equity = 0.0

        def current_equity() -> float:
            open_equity = sum(
                (last_prices[symbol] - entry[0]) * entry[2]
                for symbol, entry in entries.items()
                if symbol in last_prices
            )
            return realized_equity + open_equity

        def record_restriction(event: MarketEvent, reason: str, stage: str) -> None:
            restrictions.append({
                "timestamp": event.timestamp.isoformat(),
                "symbol": event.symbol,
                "stage": stage,
                "reason": reason,
            })

        def close_position(symbol: str, raw_exit_price: float, exit_reason: str | None = None, rollover_to: str | None = None) -> float:
            nonlocal realized_equity
            entry_price, raw_entry_price, quantity, _entry_index = entries.pop(symbol)
            exit_price = raw_exit_price * (1 - self.slippage_rate)
            gross_return = (raw_exit_price - raw_entry_price) * quantity
            commission_cost = (entry_price + exit_price) * quantity * self.commission_rate
            slippage_cost = ((entry_price - raw_entry_price) + (raw_exit_price - exit_price)) * quantity
            trade = {
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "commission_cost": commission_cost,
                "slippage_cost": slippage_cost,
                "return": gross_return - commission_cost - slippage_cost,
            }
            if exit_reason is not None:
                trade["exit_reason"] = exit_reason
            if rollover_to is not None:
                trade["rollover_to"] = rollover_to
            trades.append(trade)
            realized_equity += trade["return"]
            return trade["return"]

        for index, event in enumerate(events):
            if self.execution_policy is not None:
                allowed, reason = self.execution_policy.validate_event(event)
                if not allowed:
                    record_restriction(event, reason or "event_rejected", "market_event")
                    equity_curve.append(current_equity())
                    continue
            last_prices[event.symbol] = event.price
            if self.execution_policy is not None:
                rollover_rule = self.execution_policy.rollover_for_successor(event)
                if (
                    rollover_rule is not None
                    and rollover_rule.expiring_symbol in entries
                    and rollover_rule.successor_symbol not in entries
                    and rollover_rule.expiring_symbol in last_prices
                ):
                    old_symbol = rollover_rule.expiring_symbol
                    new_symbol = rollover_rule.successor_symbol
                    quantity = entries[old_symbol][2]
                    old_raw_price = last_prices[old_symbol]
                    close_position(old_symbol, old_raw_price, "rollover", new_symbol)
                    new_entry_price = event.price * (1 + self.slippage_rate)
                    entries[new_symbol] = (new_entry_price, event.price, quantity, index)
                    old_exit_price = old_raw_price * (1 - self.slippage_rate)
                    rollovers.append({
                        "timestamp": event.timestamp.isoformat(),
                        "from_symbol": old_symbol,
                        "to_symbol": new_symbol,
                        "quantity": quantity,
                        "old_exit_price": old_exit_price,
                        "new_entry_price": new_entry_price,
                        "commission_cost": (old_exit_price + new_entry_price) * quantity * self.commission_rate,
                        "slippage_cost": ((old_raw_price - old_exit_price) + (new_entry_price - event.price)) * quantity,
                        "research_only": True,
                    })
            signal = self.strategy.on_event(event)
            if signal is not None:
                signal_count += 1
                if self.execution_policy is not None:
                    allowed, reason = self.execution_policy.validate_signal(event, signal)
                    if not allowed:
                        record_restriction(event, reason or "signal_rejected", "signal")
                        equity_curve.append(current_equity())
                        continue
                if signal.side is OrderSide.BUY and event.symbol not in entries:
                    entries[event.symbol] = (event.price * (1 + self.slippage_rate), event.price, signal.quantity, index)
                elif signal.side is OrderSide.SELL and event.symbol in entries:
                    close_position(event.symbol, event.price)
            equity_curve.append(current_equity())
        for symbol in list(entries):
            close_position(symbol, last_prices[symbol])
        trade_count = len(trades)
        total_return = sum(trade["return"] for trade in trades)
        wins = sum(trade["return"] > 0 for trade in trades)
        win_rate = wins / len(trades) if trades else 0.0
        peak = equity_curve[0] if equity_curve else 0.0
        max_drawdown = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        return BacktestResult(
            signal_count,
            trade_count,
            total_return,
            trades,
            win_rate,
            max_drawdown,
            equity_curve,
            restrictions,
            rollovers,
        )
