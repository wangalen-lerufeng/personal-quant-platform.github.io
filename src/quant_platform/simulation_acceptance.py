"""Read-only reconciliation and evidence gate for paper trading acceptance."""

from collections import defaultdict
from datetime import datetime
from typing import Any


def _difference(actual: float, expected: float, tolerance: float) -> float:
    value = actual - expected
    return 0.0 if abs(value) <= tolerance else value


def reconcile_simulation_snapshot(
    session_snapshot: dict[str, Any],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Rebuild paper cash and positions from immutable fill evidence."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    matcher = dict(session_snapshot.get("matcher", {}))
    initial_cash = float(session_snapshot["initial_cash"])
    fills = [dict(item) for item in matcher.get("fills", [])]
    intents = {str(key): dict(value) for key, value in dict(matcher.get("intents", {})).items()}
    orders = {str(key): str(value) for key, value in dict(matcher.get("orders", {})).items()}
    actual_positions = {
        str(symbol): float(quantity)
        for symbol, quantity in dict(matcher.get("positions", {})).items()
    }

    expected_cash = initial_cash
    expected_positions: defaultdict[str, float] = defaultdict(float)
    filled_quantities: defaultdict[str, float] = defaultdict(float)
    unknown_fill_orders: list[str] = []
    for fill in fills:
        order_id = str(fill["order_id"])
        symbol = str(fill["symbol"])
        side = str(fill["side"])
        quantity = float(fill["quantity"])
        price = float(fill["price"])
        if order_id not in intents:
            unknown_fill_orders.append(order_id)
        signed_quantity = quantity if side == "buy" else -quantity
        expected_positions[symbol] += signed_quantity
        expected_cash -= signed_quantity * price
        filled_quantities[order_id] += quantity

    position_differences = {}
    for symbol in sorted(set(expected_positions) | set(actual_positions)):
        difference = _difference(
            actual_positions.get(symbol, 0.0),
            expected_positions.get(symbol, 0.0),
            tolerance,
        )
        if difference:
            position_differences[symbol] = difference

    order_mismatches = {}
    for order_id, state in orders.items():
        intent = intents.get(order_id)
        if intent is None:
            order_mismatches[order_id] = "missing_intent"
            continue
        requested = float(intent["quantity"])
        filled = filled_quantities.get(order_id, 0.0)
        if filled > requested + tolerance:
            order_mismatches[order_id] = "overfilled"
        elif state == "filled" and abs(filled - requested) > tolerance:
            order_mismatches[order_id] = "filled_state_quantity_mismatch"
        elif state == "partially_filled" and not tolerance < filled < requested - tolerance:
            order_mismatches[order_id] = "partial_state_quantity_mismatch"

    actual_cash = float(matcher.get("cash", 0.0))
    cash_difference = _difference(actual_cash, expected_cash, tolerance)
    discrepancies = []
    if cash_difference:
        discrepancies.append("cash_mismatch")
    if position_differences:
        discrepancies.append("position_mismatch")
    if order_mismatches:
        discrepancies.append("order_state_mismatch")
    if unknown_fill_orders:
        discrepancies.append("unknown_fill_order")
    return {
        "passed": not discrepancies,
        "fill_count": len(fills),
        "filled_quantities": dict(filled_quantities),
        "checks": {
            "cash": {
                "expected": expected_cash,
                "actual": actual_cash,
                "difference": cash_difference,
            },
            "positions": {
                "expected": dict(expected_positions),
                "actual": actual_positions,
                "differences": position_differences,
            },
            "orders": {
                "order_count": len(orders),
                "mismatches": order_mismatches,
                "unknown_fill_orders": sorted(set(unknown_fill_orders)),
            },
        },
        "discrepancies": discrepancies,
        "paper_only": True,
        "execution_enabled": False,
    }


class SimulationAcceptanceGate:
    """Require sustained paper evidence before a human may review live readiness."""

    def __init__(self, minimum_trading_days: int = 30):
        if minimum_trading_days <= 0:
            raise ValueError("minimum_trading_days must be positive")
        self.minimum_trading_days = minimum_trading_days

    def assess(
        self,
        session_snapshot: dict[str, Any],
        account_snapshots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trading_dates = set()
        invalid_snapshot_count = 0
        for snapshot in account_snapshots:
            try:
                timestamp = datetime.fromisoformat(str(snapshot["created_at"]))
                trading_dates.add(timestamp.date().isoformat())
            except (KeyError, TypeError, ValueError):
                invalid_snapshot_count += 1

        reconciliation = reconcile_simulation_snapshot(session_snapshot)
        blockers = []
        if len(trading_dates) < self.minimum_trading_days:
            blockers.append("minimum_paper_trading_days_not_met")
        if not reconciliation["passed"]:
            blockers.append("reconciliation_failed")
        if bool(session_snapshot.get("halted", False)):
            blockers.append("account_risk_halted")
        operational = dict(session_snapshot.get("operational_risk", {}))
        manual_alert = operational.get("manual_alert")
        loss_limit = operational.get("max_consecutive_losses")
        api_limit = operational.get("max_api_failures")
        if manual_alert:
            blockers.append("manual_kill_switch_active")
        if loss_limit is not None and int(operational.get("consecutive_losses", 0)) >= int(loss_limit):
            blockers.append("consecutive_loss_limit")
        if api_limit is not None and int(operational.get("api_failures", 0)) >= int(api_limit):
            blockers.append("api_failure_limit")

        eligible = not blockers
        return {
            "status": "eligible_for_manual_review" if eligible else "blocked",
            "eligible_for_manual_review": eligible,
            "paper_trading_days": len(trading_dates),
            "minimum_trading_days": self.minimum_trading_days,
            "trading_dates": sorted(trading_dates),
            "invalid_snapshot_count": invalid_snapshot_count,
            "reconciliation": reconciliation,
            "blockers": blockers,
            "live_trading_enabled": False,
            "execution_enabled": False,
        }
