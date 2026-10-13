import unittest

from quant_platform.live_readiness import LiveReadinessEvidence, LiveReadinessGate


class LiveReadinessTests(unittest.TestCase):
    def test_missing_evidence_blocks_live_request(self):
        result = LiveReadinessGate(minimum_paper_days=30).assess(
            LiveReadinessEvidence(
                out_of_sample_passed=False,
                paper_trading_days=10,
                reconciliation_passed=False,
                risk_limits_configured=False,
                manual_approval_id=None,
            )
        )

        self.assertFalse(result["eligible_to_request"])
        self.assertIn("out_of_sample_test_required", result["blockers"])
        self.assertIn("minimum_paper_trading_days_not_met", result["blockers"])
        self.assertIn("manual_approval_required", result["blockers"])
        self.assertFalse(result["execution_enabled"])

    def test_complete_evidence_only_allows_manual_review_not_execution(self):
        result = LiveReadinessGate(minimum_paper_days=30).assess(
            LiveReadinessEvidence(
                out_of_sample_passed=True,
                paper_trading_days=45,
                reconciliation_passed=True,
                risk_limits_configured=True,
                manual_approval_id="approval-001",
            )
        )

        self.assertTrue(result["eligible_to_request"])
        self.assertEqual(result["status"], "eligible_for_manual_review")
        self.assertFalse(result["live_trading_enabled"])
        self.assertFalse(result["execution_enabled"])


if __name__ == "__main__":
    unittest.main()
