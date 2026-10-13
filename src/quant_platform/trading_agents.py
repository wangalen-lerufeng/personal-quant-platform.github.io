"""TradingAgents-style read-only evidence verification for Phase 9.

Deterministic multi-role verification: fundamental, technical, news, risk, and
adversarial (bearish) roles each examine read-only evidence and emit a stance,
then a synthesizer aggregates them into a structured 通过/观察/排除 decision with
risk tags, evidence citations, and conflicting opinions.

The core is intentionally deterministic and model-free so it is auditable and
testable; a model layer can produce per-role evidence that feeds the same
synthesis. No orders, no risk-engine mutation — every output is ``research_only``
/ ``execution_enabled: false``.
"""

from dataclasses import dataclass
from typing import Any

ROLES = ("fundamental", "technical", "news", "risk", "bearish")

HARD_EXCLUDE_FIELDS = frozenset({"suspended", "delisted", "kill_switch"})
SOFT_OBSERVE_FIELDS = frozenset({
    "impairment",
    "cash_flow_quality_low",
    "high_volatility",
    "high_concentration",
    "negative_news",
    "liquidity_warning",
    "stale_data",
})


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    dataset: str
    as_of: str
    field: str
    value: Any
    note: str = ""


@dataclass(frozen=True)
class RoleVerdict:
    role: str
    stance: str  # "pass" | "observe" | "exclude"
    risk_tags: tuple[str, ...] = ()
    rationale: str = ""
    evidence: tuple[EvidenceItem, ...] = ()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "stance": self.stance,
            "risk_tags": list(self.risk_tags),
            "rationale": self.rationale,
            "evidence": [_evidence_to_dict(item) for item in self.evidence],
        }


@dataclass(frozen=True)
class CandidateDecision:
    symbol: str
    verdict: str  # "pass" | "observe" | "exclude"
    risk_tags: tuple[str, ...]
    roles: tuple[RoleVerdict, ...]
    conflicts: tuple[str, ...]
    verified: bool

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "verdict": self.verdict,
            "risk_tags": list(self.risk_tags),
            "roles": [role.to_dict() for role in self.roles],
            "conflicts": list(self.conflicts),
            "verified": self.verified,
            "research_only": True,
            "execution_enabled": False,
        }


def _evidence_to_dict(item: EvidenceItem) -> dict:
    return {
        "source": item.source,
        "dataset": item.dataset,
        "as_of": item.as_of,
        "field": item.field,
        "value": item.value,
        "note": item.note,
    }


def verdict_for_role(role: str, items: list[EvidenceItem]) -> RoleVerdict:
    """Assign a stance to one role from its evidence.

    No evidence -> observe (原因未验证). Hard flag -> exclude. Soft flag ->
    observe. Otherwise pass.
    """
    if not items:
        return RoleVerdict(role, "observe", ("原因未验证",), f"{role} 缺少证据", ())
    hard = [item for item in items if item.field in HARD_EXCLUDE_FIELDS]
    soft = [item for item in items if item.field in SOFT_OBSERVE_FIELDS]
    if hard:
        tags = tuple(item.note or item.field for item in hard)
        return RoleVerdict(role, "exclude", tags, f"{len(hard)} 项硬性风险", tuple(items))
    if soft:
        tags = tuple(item.note or item.field for item in soft)
        return RoleVerdict(role, "observe", tags, f"{len(soft)} 项观察项", tuple(items))
    return RoleVerdict(role, "pass", (), "证据无异常", tuple(items))


def build_role_verdicts(evidence_items: list[EvidenceItem]) -> tuple[RoleVerdict, ...]:
    """Route evidence by source and run each role's verification."""
    by_source: dict[str, list[EvidenceItem]] = {}
    for item in evidence_items:
        by_source.setdefault(item.source, []).append(item)
    return tuple(verdict_for_role(role, by_source.get(role, [])) for role in ROLES)


def synthesize_decision(symbol: str, role_verdicts: tuple[RoleVerdict, ...]) -> CandidateDecision:
    """Aggregate role verdicts into a structured candidate decision."""
    if any(verdict.stance == "exclude" for verdict in role_verdicts):
        verdict = "exclude"
    elif any(verdict.stance == "observe" for verdict in role_verdicts):
        verdict = "observe"
    else:
        verdict = "pass"

    risk_tags = tuple(sorted({tag for role in role_verdicts for tag in role.risk_tags}))

    conflicts: list[str] = []
    roles = list(role_verdicts)
    for index, first in enumerate(roles):
        for second in roles[index + 1:]:
            if first.stance != second.stance:
                conflicts.append(f"{first.role}_vs_{second.role}")

    verified = all(bool(role.evidence) for role in role_verdicts)
    return CandidateDecision(symbol, verdict, risk_tags, tuple(roles), tuple(conflicts), verified)
