import unittest

import numpy as np

from quant_platform.position_sizing import (
    SizingCosts,
    build_states,
    clamp_adjustment,
    evaluate_policy,
    identity_sizer,
    should_fallback,
    simulate,
    volatility_target_sizer,
)


class ClampTests(unittest.TestCase):
    def test_clamps_to_band(self):
        self.assertEqual(clamp_adjustment(1.5), 1.2)
        self.assertEqual(clamp_adjustment(0.5), 0.8)
        self.assertEqual(clamp_adjustment(1.0), 1.0)


class BuildStatesTests(unittest.TestCase):
    def test_states_are_causal(self):
        returns = np.array([0.1, 0.2, 0.3])
        baseline = np.array([1.0, 1.0, 1.0])
        states = build_states(returns, baseline, window=2)
        self.assertEqual(states.shape, (3, 3))
        self.assertTrue(np.allclose(states[0], [1.0, 0.0, 0.0]))
        self.assertAlmostEqual(states[1][1], 0.1)
        self.assertAlmostEqual(states[1][2], 0.0)
        self.assertAlmostEqual(states[2][1], 0.15)


class SimulateTests(unittest.TestCase):
    def test_identity_adjustment_zero_costs(self):
        returns = np.array([0.1, -0.1, 0.05])
        baseline = np.ones(3)
        result = simulate(returns, baseline, np.ones(3), SizingCosts())
        self.assertAlmostEqual(result.gross_return, 0.05)
        self.assertAlmostEqual(result.turnover_cost, 0.0)
        self.assertAlmostEqual(result.total_reward, 0.05)

    def test_turnover_cost_is_charged(self):
        returns = np.array([0.1, -0.1, 0.05])
        baseline = np.ones(3)
        adjustments = np.array([1.2, 0.8, 1.2])
        costs = SizingCosts(cost_rate=0.1)
        result = simulate(returns, baseline, adjustments, costs)
        self.assertAlmostEqual(result.gross_return, 0.10)
        self.assertAlmostEqual(result.total_turnover, 1.0)
        self.assertAlmostEqual(result.turnover_cost, 0.1)

    def test_risk_and_drawdown_penalties(self):
        returns = np.array([0.1, -0.1, 0.05])
        baseline = np.ones(3)
        costs = SizingCosts(risk_exposure_penalty=1.0, drawdown_penalty=1.0)
        result = simulate(returns, baseline, np.ones(3), costs)
        self.assertGreater(result.risk_penalty, 0.0)
        self.assertGreater(result.drawdown_penalty, 0.0)
        self.assertGreater(result.max_drawdown, 0.0)


class EvaluatePolicyTests(unittest.TestCase):
    def test_identity_matches_simulate(self):
        returns = np.array([0.1, -0.1, 0.05])
        baseline = np.ones(3)
        via_policy = evaluate_policy(identity_sizer, returns, baseline, SizingCosts())
        direct = simulate(returns, baseline, np.ones(3), SizingCosts())
        self.assertAlmostEqual(via_policy.total_reward, direct.total_reward)


class VolatilityTargetTests(unittest.TestCase):
    def test_volatility_target_clamps_to_band(self):
        sizer = volatility_target_sizer(vol_target=0.2)
        self.assertEqual(sizer(np.array([1.0, 0.0, 0.4])), 0.8)
        self.assertEqual(sizer(np.array([1.0, 0.0, 0.1])), 1.2)
        self.assertEqual(sizer(np.array([1.0, 0.0, 0.0])), 1.0)


class FallbackTests(unittest.TestCase):
    def test_fallback_when_candidate_worse(self):
        returns = np.array([0.1, -0.1, 0.05])
        baseline = np.ones(3)
        identity = simulate(returns, baseline, np.ones(3), SizingCosts(cost_rate=0.1))
        worse = simulate(returns, baseline, np.array([1.2, 0.8, 1.2]), SizingCosts(cost_rate=0.1))
        self.assertTrue(should_fallback(worse, identity))


if __name__ == "__main__":
    unittest.main()
