"""Read-only readiness assessment for future live-trading review."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveReadinessEvidence:
    out_of_sample_passed: bool
    paper_trading_days: int
    reconciliation_passed: bool
    risk_limits_configured: bool
    manual_approval_id: str | None = None

    def __post_init__(self):
        if self.paper_trading_days < 0:
            raise ValueError("paper_trading_days must be non-negative")


class LiveReadinessGate:
    """Evaluates evidence but has no authority to enable live execution."""

    def __init__(self, minimum_paper_days: int = 30):
        if minimum_paper_days <= 0:
            raise ValueError("minimum_paper_days must be positive")
        self.minimum_paper_days = minimum_paper_days

    def assess(self, evidence: LiveReadinessEvidence) -> dict[str, object]:
        blockers: list[str] = []
        if not evidence.out_of_sample_passed:
            blockers.append("out_of_sample_test_required")
        if evidence.paper_trading_days < self.minimum_paper_days:
            blockers.append("minimum_paper_trading_days_not_met")
        if not evidence.reconciliation_passed:
            blockers.append("order_reconciliation_required")
        if not evidence.risk_limits_configured:
            blockers.append("risk_limits_required")
        if not (evidence.manual_approval_id or "").strip():
            blockers.append("manual_approval_required")
        eligible = not blockers
        return {
            "status": "eligible_for_manual_review" if eligible else "blocked",
            "eligible_to_request": eligible,
            "blockers": blockers,
            "minimum_paper_days": self.minimum_paper_days,
            "paper_trading_days": evidence.paper_trading_days,
            "manual_approval_id": evidence.manual_approval_id,
            "live_trading_enabled": False,
            "execution_enabled": False,
        }
