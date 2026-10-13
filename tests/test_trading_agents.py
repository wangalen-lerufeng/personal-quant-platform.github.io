import unittest

from quant_platform.trading_agents import (
    EvidenceItem,
    RoleVerdict,
    build_role_verdicts,
    synthesize_decision,
    verdict_for_role,
)


def ev(source, field, note="", value=True):
    return EvidenceItem(source, "dataset", "2026-01-05", field, value, note)


def _verdict(role, stance, tags=(), with_evidence=True):
    evidence = (ev(role, "field"),) if with_evidence else ()
    return RoleVerdict(role, stance, tags, "", evidence)


class VerdictForRoleTests(unittest.TestCase):
    def test_missing_evidence_is_observe_and_unverified(self):
        verdict = verdict_for_role("fundamental", [])
        self.assertEqual(verdict.stance, "observe")
        self.assertIn("原因未验证", verdict.risk_tags)

    def test_hard_flag_excludes(self):
        verdict = verdict_for_role("fundamental", [ev("fundamental", "suspended", "停牌")])
        self.assertEqual(verdict.stance, "exclude")
        self.assertIn("停牌", verdict.risk_tags)

    def test_soft_flag_observes(self):
        verdict = verdict_for_role("fundamental", [ev("fundamental", "impairment", "减值")])
        self.assertEqual(verdict.stance, "observe")
        self.assertIn("减值", verdict.risk_tags)

    def test_clean_evidence_passes(self):
        verdict = verdict_for_role("fundamental", [ev("fundamental", "net_profit", "净利润为正")])
        self.assertEqual(verdict.stance, "pass")
        self.assertEqual(verdict.risk_tags, ())


class BuildRoleVerdictsTests(unittest.TestCase):
    def test_routes_evidence_by_source(self):
        items = [
            ev("fundamental", "net_profit"),
            ev("technical", "close"),
            ev("news", "anns_d"),
            ev("risk", "drawdown"),
        ]
        verdicts = build_role_verdicts(items)
        roles = [verdict.role for verdict in verdicts]
        self.assertEqual(roles, ["fundamental", "technical", "news", "risk", "bearish"])
        by_role = {verdict.role: verdict for verdict in verdicts}
        self.assertEqual(by_role["fundamental"].stance, "pass")
        self.assertEqual(by_role["technical"].stance, "pass")
        self.assertEqual(by_role["bearish"].stance, "observe")


class SynthesizeTests(unittest.TestCase):
    def test_exclude_dominates(self):
        roles = (
            _verdict("fundamental", "pass"),
            _verdict("technical", "pass"),
            _verdict("news", "exclude", ("停牌",)),
            _verdict("risk", "pass"),
            _verdict("bearish", "pass"),
        )
        decision = synthesize_decision("510300.SH", roles)
        self.assertEqual(decision.verdict, "exclude")
        self.assertIn("停牌", decision.risk_tags)

    def test_observe_when_any_observe(self):
        roles = (
            _verdict("fundamental", "pass"),
            _verdict("technical", "observe", ("高波动",)),
            _verdict("news", "pass"),
            _verdict("risk", "pass"),
            _verdict("bearish", "pass"),
        )
        decision = synthesize_decision("510300.SH", roles)
        self.assertEqual(decision.verdict, "observe")
        self.assertIn("高波动", decision.risk_tags)

    def test_pass_requires_all_pass(self):
        roles = tuple(_verdict(role, "pass") for role in (
            "fundamental", "technical", "news", "risk", "bearish",
        ))
        decision = synthesize_decision("510300.SH", roles)
        self.assertEqual(decision.verdict, "pass")
        self.assertEqual(decision.risk_tags, ())
        self.assertTrue(decision.verified)

    def test_collects_risk_tags_and_conflicts(self):
        roles = (
            _verdict("fundamental", "pass"),
            _verdict("technical", "observe", ("高波动",)),
            _verdict("news", "pass"),
            _verdict("risk", "observe", ("流动性低",)),
            _verdict("bearish", "pass"),
        )
        decision = synthesize_decision("510300.SH", roles)
        self.assertEqual(set(decision.risk_tags), {"高波动", "流动性低"})
        self.assertTrue(any("technical" in conflict for conflict in decision.conflicts))

    def test_verified_requires_evidence_for_every_role(self):
        roles = (
            _verdict("fundamental", "pass", with_evidence=True),
            _verdict("technical", "pass", with_evidence=True),
            _verdict("news", "pass", with_evidence=True),
            _verdict("risk", "pass", with_evidence=True),
            _verdict("bearish", "pass", with_evidence=False),
        )
        decision = synthesize_decision("510300.SH", roles)
        self.assertFalse(decision.verified)

    def test_to_dict_marks_research_only(self):
        roles = tuple(_verdict(role, "pass") for role in (
            "fundamental", "technical", "news", "risk", "bearish",
        ))
        payload = synthesize_decision("510300.SH", roles).to_dict()
        self.assertFalse(payload["execution_enabled"])
        self.assertTrue(payload["research_only"])
        self.assertEqual(payload["verdict"], "pass")
        self.assertEqual(len(payload["roles"]), 5)


if __name__ == "__main__":
    unittest.main()
