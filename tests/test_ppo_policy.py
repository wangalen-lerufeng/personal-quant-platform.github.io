import unittest

import numpy as np

from quant_platform.position_sizing import SizingCosts, evaluate_policy
from quant_platform.ppo_policy import PPOPolicy, train_ppo


def _block_returns():
    return np.array([0.5] * 50 + [-0.5] * 50 + [0.5] * 50 + [-0.5] * 50)


class PPOPolicyTests(unittest.TestCase):
    def test_predict_respects_band(self):
        policy = PPOPolicy(state_dim=3, seed=0)
        for state in ([1.0, 1.0, 1.0], [1.0, -1.0, 0.5], [0.5, 0.0, 0.0]):
            adjustment = policy.predict(np.array(state))
            self.assertGreaterEqual(adjustment, 0.8)
            self.assertLessEqual(adjustment, 1.2)

    def test_training_is_deterministic(self):
        returns = _block_returns()
        baseline = np.ones_like(returns)
        first = train_ppo(returns, baseline, costs=SizingCosts(), window=10, episodes=30, seed=7)
        second = train_ppo(returns, baseline, costs=SizingCosts(), window=10, episodes=30, seed=7)
        probe = np.array([1.0, 0.02, 0.0])
        self.assertAlmostEqual(first.predict(probe), second.predict(probe))

    def test_learns_to_raise_exposure_in_uptrend(self):
        returns = _block_returns()
        baseline = np.ones_like(returns)
        policy = train_ppo(
            returns, baseline, costs=SizingCosts(), window=10, episodes=500, seed=0, lr=0.2,
        )
        up = policy.predict(np.array([1.0, 0.5, 0.0]))
        down = policy.predict(np.array([1.0, -0.5, 0.0]))
        self.assertGreater(up, down)

    def test_trained_policy_evaluates(self):
        returns = _block_returns()
        baseline = np.ones_like(returns)
        policy = train_ppo(
            returns, baseline, costs=SizingCosts(), window=10, episodes=50, seed=0,
        )
        result = evaluate_policy(policy.predict, returns, baseline, SizingCosts())
        self.assertIsInstance(result.total_reward, float)


if __name__ == "__main__":
    unittest.main()
