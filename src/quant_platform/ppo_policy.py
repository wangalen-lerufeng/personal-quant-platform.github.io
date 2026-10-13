"""Minimal numpy PPO policy for constrained position sizing (Phase 9 step5).

A small clipped-surrogate PPO with a linear-Gaussian policy and a linear value
function, trained with manual gradients so no deep-learning framework (torch) is
required. The action is a position-adjustment factor clamped to [0.8, 1.2] by
the sizing environment; the policy's own mean is squashed to the same band.

Read-only research: the policy only proposes adjustment factors and never
touches orders or risk limits.
"""

import numpy as np

from .position_sizing import (
    ADJUSTMENT_HIGH,
    ADJUSTMENT_LOW,
    build_states,
    per_step_rewards,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _gaussian_log_prob(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2 * sigma ** 2)


def _discount(rewards, gamma):
    length = len(rewards)
    discounted = np.zeros(length, dtype=float)
    running = 0.0
    for index in reversed(range(length)):
        running = rewards[index] + gamma * running
        discounted[index] = running
    return discounted


class PPOPolicy:
    def __init__(
        self,
        state_dim: int,
        *,
        low: float = ADJUSTMENT_LOW,
        high: float = ADJUSTMENT_HIGH,
        lr: float = 0.1,
        clip: float = 0.2,
        sigma: float = 0.3,
        seed: int = 0,
        c_value: float = 0.5,
        epochs: int = 5,
    ):
        self.low = low
        self.high = high
        self.lr = lr
        self.clip = clip
        self.sigma = sigma
        self.c_value = c_value
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(0.0, 0.05, size=state_dim)
        self.b = 0.0
        self.value_w = np.zeros(state_dim)
        self.value_b = 0.0

    def _mean(self, states):
        z = states @ self.w + self.b
        return self.low + (self.high - self.low) * _sigmoid(z)

    def value(self, states):
        return states @ self.value_w + self.value_b

    def predict(self, state) -> float:
        state = np.asarray(state, dtype=float)
        return float(self._mean(state[None, :])[0])

    def sample(self, states):
        mu = self._mean(states)
        raw = mu + self.rng.normal(0.0, self.sigma, size=mu.shape)
        log_probs = _gaussian_log_prob(raw, mu, self.sigma)
        return raw, log_probs

    def update(self, states, raw_actions, old_log_probs, advantages, returns_disc):
        length = len(states)
        for _ in range(self.epochs):
            mu = self._mean(states)
            new_log_probs = _gaussian_log_prob(raw_actions, mu, self.sigma)
            ratio = np.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * advantages
            grad_mask = unclipped <= clipped
            dloss_dlogprob = -np.where(grad_mask, ratio * advantages, 0.0)

            z = states @ self.w + self.b
            sig = _sigmoid(z)
            dmu_dz = (self.high - self.low) * sig * (1 - sig)
            dlogprob_dz = (raw_actions - mu) / (self.sigma ** 2) * dmu_dz

            dpolicy_w = (dloss_dlogprob[:, None] * dlogprob_dz[:, None] * states).mean(axis=0)
            dpolicy_b = (dloss_dlogprob * dlogprob_dz).mean()

            values = self.value(states)
            dvalue = (2.0 / length) * (values - returns_disc)
            dvalue_w = dvalue @ states
            dvalue_b = dvalue.sum()

            self.w -= self.lr * dpolicy_w
            self.b -= self.lr * dpolicy_b
            self.value_w -= self.lr * self.c_value * dvalue_w
            self.value_b -= self.lr * self.c_value * dvalue_b


def train_ppo(
    returns,
    baseline_weights,
    *,
    costs=None,
    window: int = 20,
    episodes: int = 200,
    gamma: float = 0.99,
    **policy_kwargs,
) -> PPOPolicy:
    """Train a PPO sizing policy on a returns series and return it."""
    returns = np.asarray(returns, dtype=float)
    baseline_weights = np.asarray(baseline_weights, dtype=float)
    if costs is None:
        from .position_sizing import SizingCosts

        costs = SizingCosts()
    states = build_states(returns, baseline_weights, window)
    policy = PPOPolicy(states.shape[1], **policy_kwargs)
    for _ in range(episodes):
        raw_actions, log_probs = policy.sample(states)
        clipped_actions = np.clip(raw_actions, ADJUSTMENT_LOW, ADJUSTMENT_HIGH)
        rewards = per_step_rewards(returns, baseline_weights, clipped_actions, costs)
        returns_disc = _discount(rewards, gamma)
        advantages = returns_disc - policy.value(states)
        policy.update(states, raw_actions, log_probs, advantages, returns_disc)
    return policy
