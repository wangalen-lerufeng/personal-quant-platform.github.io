"""Paper-only simulation session that links matching, account state and risk."""

from dataclasses import asdict

from .domain import OrderIntent
from .execution import PaperMatchingEngine
from .risk import OperationalRiskGuard, RiskEngine, RiskLimits, RiskMonitor


class PaperSimulationSession:
    """Coordinate deterministic paper fills without exposing a live gateway."""

    def __init__(
        self,
        account_id: str,
        initial_cash: float,
        account_store=None,
        risk_limits: RiskLimits | None = None,
        operational_guard: OperationalRiskGuard | None = None,
        latency_events: int = 0,
        max_participation_rate: float = 0.1,
        slippage_rate: float = 0.0,
    ):
        if not account_id.strip():
            raise ValueError("account_id is required")
        self.account_id = account_id
        self.account_store = account_store
        self.initial_cash = float(initial_cash)
        self.limits = risk_limits or RiskLimits()
        self.matcher = PaperMatchingEngine(
            initial_cash=initial_cash,
            latency_events=latency_events,
            max_participation_rate=max_participation_rate,
            slippage_rate=slippage_rate,
        )
        self.order_risk = RiskEngine(self.limits)
        self.account_risk = RiskMonitor(self.limits, initial_cash)
        self.operational_guard = operational_guard or OperationalRiskGuard()
        self.latest_prices: dict[str, float] = {}
        self.latest_risk = self.account_risk.observe(initial_cash)
        self.halted = False

    def submit(self, intent: OrderIntent) -> str:
        if self.halted:
            raise RuntimeError("simulation_halted")
        operational = self.operational_guard.status()
        if operational["halted"]:
            raise RuntimeError(
                "simulation_halted:" + ",".join(operational["alerts"])
            )
        decision = self.order_risk.evaluate(intent)
        if not decision.allowed:
            raise RuntimeError(f"risk_rejected:{decision.reason}")
        order_id = self.matcher.submit(intent)
        self._persist()
        return order_id

    def on_market(self, symbol: str, price: float, volume: float) -> dict[str, object]:
        self.operational_guard.record_market_event()
        self.latest_prices[symbol] = float(price)
        fills = self.matcher.on_market(symbol, price, volume)
        for fill in fills:
            self.order_risk.record_fill(
                fill.symbol,
                self.matcher.intent_for(fill.order_id).side,
                fill.quantity,
            )
        equity = self.matcher.cash + sum(
            quantity * self.latest_prices.get(position_symbol, 0.0)
            for position_symbol, quantity in self.matcher.positions.items()
        )
        self.latest_risk = self.account_risk.observe(equity)
        self.halted = self.latest_risk["risk_level"] == "halt"
        if self.account_store is not None:
            self.account_store.save_snapshot(
                self.account_id,
                self.matcher.cash,
                equity,
                self.matcher.positions,
            )
        self._persist()
        return {
            "account_id": self.account_id,
            "fills": [asdict(fill) for fill in fills],
            "orders": {
                order_id: lifecycle.state.value
                for order_id, lifecycle in self.matcher.orders.items()
            },
            "account": {
                "cash": self.matcher.cash,
                "equity": equity,
                "positions": dict(self.matcher.positions),
                "latest_prices": dict(self.latest_prices),
            },
            "risk": dict(self.latest_risk),
            "operational_risk": self.operational_guard.status(),
            "paper_only": True,
            "execution_enabled": False,
        }

    def state(self) -> dict[str, object]:
        equity = self.matcher.cash + sum(
            quantity * self.latest_prices.get(symbol, 0.0)
            for symbol, quantity in self.matcher.positions.items()
        )
        return {
            "account_id": self.account_id,
            "account": {
                "cash": self.matcher.cash,
                "equity": equity,
                "positions": dict(self.matcher.positions),
                "latest_prices": dict(self.latest_prices),
            },
            "orders": {
                order_id: lifecycle.state.value
                for order_id, lifecycle in self.matcher.orders.items()
            },
            "risk": dict(self.latest_risk),
            "operational_risk": self.operational_guard.status(),
            "paper_only": True,
            "execution_enabled": False,
        }

    def record_trade_pnl(self, pnl: float) -> dict[str, object]:
        self.operational_guard.record_trade_pnl(pnl)
        self._persist()
        return self.operational_guard.status()

    def record_api_result(self, success: bool) -> dict[str, object]:
        self.operational_guard.record_api_result(success)
        self._persist()
        return self.operational_guard.status()

    def activate_kill_switch(self, reason: str = "manual_kill_switch") -> dict[str, object]:
        self.operational_guard.activate_kill_switch(reason)
        self._persist()
        return self.operational_guard.status()

    def snapshot(self) -> dict[str, object]:
        return {
            "version": 1,
            "account_id": self.account_id,
            "initial_cash": self.initial_cash,
            "risk_limits": asdict(self.limits),
            "matcher": self.matcher.snapshot(),
            "latest_prices": dict(self.latest_prices),
            "account_risk": {
                "day_start_equity": self.account_risk.day_start_equity,
                "peak_equity": self.account_risk.peak_equity,
                "current_equity": self.account_risk.current_equity,
            },
            "operational_risk": self.operational_guard.snapshot(),
            "halted": self.halted,
        }

    def _persist(self) -> None:
        if self.account_store is not None and hasattr(
            self.account_store, "save_session_state"
        ):
            self.account_store.save_session_state(self.account_id, self.snapshot())

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object], account_store=None):
        matcher_snapshot = dict(snapshot["matcher"])
        limits = RiskLimits(**dict(snapshot.get("risk_limits", {})))
        operational = OperationalRiskGuard.from_snapshot(
            dict(snapshot.get("operational_risk", {}))
        )
        session = cls(
            account_id=str(snapshot["account_id"]),
            initial_cash=float(snapshot["initial_cash"]),
            account_store=account_store,
            risk_limits=limits,
            operational_guard=operational,
            latency_events=int(matcher_snapshot.get("latency_events", 0)),
            max_participation_rate=float(
                matcher_snapshot.get("max_participation_rate", 0.1)
            ),
            slippage_rate=float(matcher_snapshot.get("slippage_rate", 0.0)),
        )
        session.matcher = PaperMatchingEngine.from_snapshot(matcher_snapshot)
        session.order_risk.positions = dict(session.matcher.positions)
        session.latest_prices = {
            str(symbol): float(price)
            for symbol, price in dict(snapshot.get("latest_prices", {})).items()
        }
        risk_state = dict(snapshot.get("account_risk", {}))
        session.account_risk.day_start_equity = float(
            risk_state.get("day_start_equity", session.initial_cash)
        )
        session.account_risk.peak_equity = float(
            risk_state.get("peak_equity", session.initial_cash)
        )
        restored_equity = float(
            risk_state.get("current_equity", session.initial_cash)
        )
        session.latest_risk = session.account_risk.observe(restored_equity)
        session.halted = bool(snapshot.get("halted", False)) or (
            session.latest_risk["risk_level"] == "halt"
        )
        return session


class PaperSimulationRegistry:
    """Owns isolated paper sessions for the local control console."""

    def __init__(self, account_store=None):
        self.account_store = account_store
        self._sessions: dict[str, PaperSimulationSession] = {}
        if account_store is not None and hasattr(account_store, "list_session_states"):
            for snapshot in account_store.list_session_states():
                session = PaperSimulationSession.from_snapshot(snapshot, account_store)
                self._sessions[session.account_id] = session

    def create(
        self,
        account_id: str,
        initial_cash: float,
        risk_limits: RiskLimits | None = None,
        max_consecutive_losses: int | None = None,
        market_data_timeout_seconds: float | None = None,
        max_api_failures: int | None = None,
        latency_events: int = 0,
        max_participation_rate: float = 0.1,
        slippage_rate: float = 0.0,
    ) -> PaperSimulationSession:
        if account_id in self._sessions:
            raise ValueError(f"simulation account already exists: {account_id}")
        session = PaperSimulationSession(
            account_id=account_id,
            initial_cash=initial_cash,
            account_store=self.account_store,
            risk_limits=risk_limits,
            operational_guard=OperationalRiskGuard(
                max_consecutive_losses=max_consecutive_losses,
                market_data_timeout_seconds=market_data_timeout_seconds,
                max_api_failures=max_api_failures,
            ),
            latency_events=latency_events,
            max_participation_rate=max_participation_rate,
            slippage_rate=slippage_rate,
        )
        self._sessions[account_id] = session
        session._persist()
        return session

    def get(self, account_id: str) -> PaperSimulationSession:
        try:
            return self._sessions[account_id]
        except KeyError as error:
            raise KeyError(f"unknown simulation account: {account_id}") from error

    def delete(self, account_id: str) -> dict[str, object]:
        if account_id not in self._sessions:
            raise KeyError(f"unknown simulation account: {account_id}")
        del self._sessions[account_id]
        if self.account_store is not None and hasattr(self.account_store, "delete_account"):
            self.account_store.delete_account(account_id)
        return {"deleted": account_id, "paper_only": True, "execution_enabled": False}

    def list_states(self) -> list[dict[str, object]]:
        return [session.state() for session in self._sessions.values()]
