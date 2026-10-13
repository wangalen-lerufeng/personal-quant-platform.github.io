"""Constrained position-sizing environment for Phase 9 step5.

Deterministic, no-look-ahead simulation of a baseline position adjusted by a
factor clamped to [0.8, 1.2]. The reward deducts turnover cost, risk exposure,
and drawdown. Provides fixed/equal-weight and volatility-targeted baselines plus
a fallback that returns to baseline when a candidate underperforms.

Pure numpy (no torch). A sizing policy maps a causal state to an adjustment
factor; the PPO policy in ``ppo_policy`` implements the same contract.
"""

from dataclasses import dataclass

import numpy as np

ADJUSTMENT_LOW = 0.8
ADJUSTMENT_HIGH = 1.2


@dataclass(frozen=True)
class SizingCosts:
    cost_rate: float = 0.001
    risk_exposure_penalty: float = 0.0
    drawdown_penalty: float = 0.0


@dataclass(frozen=True)
class SizingResult:
    total_reward: float
    gross_return: float
    turnover_cost: float
    risk_penalty: float
    drawdown_penalty: float
    max_drawdown: float
    total_turnover: float
    equity_curve: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "total_reward": self.total_reward,
            "gross_return": self.gross_return,
            "turnover_cost": self.turnover_cost,
            "risk_penalty": self.risk_penalty,
            "drawdown_penalty": self.drawdown_penalty,
            "max_drawdown": self.max_drawdown,
            "total_turnover": self.total_turnover,
            "equity_curve": list(self.equity_curve),
            "research_only": True,
            "execution_enabled": False,
        }


def clamp_adjustment(
    value: float,
    low: float = ADJUSTMENT_LOW,
    high: float = ADJUSTMENT_HIGH,
) -> float:
    return float(min(max(value, low), high))


def build_states(
    returns: np.ndarray,
    baseline_weights: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Build causal per-step states [baseline, momentum, volatility].

    The state at step ``t`` uses only ``returns[:t]`` (strictly before ``t``),
    so no future data leaks into the decision.
    """
    returns = np.asarray(returns, dtype=float)
    baseline = np.asarray(baseline_weights, dtype=float)
    if returns.ndim != 1 or baseline.ndim != 1:
        raise ValueError("returns and baseline_weights must be 1-D")
    if len(returns) != len(baseline):
        raise ValueError("returns and baseline_weights must have the same length")
    states = np.zeros((len(returns), 3), dtype=float)
    for index in range(len(returns)):
        history = returns[:index]
        momentum = float(np.mean(history[-window:])) if len(history) else 0.0
        volatility = float(np.std(history[-window:])) if len(history) >= 2 else 0.0
        states[index] = [baseline[index], momentum, volatility]
    return states


def _simulate(returns, baseline_weights, adjustments, costs):
    returns = np.asarray(returns, dtype=float)
    baseline = np.asarray(baseline_weights, dtype=float)
    adjustments = np.asarray(adjustments, dtype=float)
    if not (len(returns) == len(baseline) == len(adjustments)):
        raise ValueError("returns, baseline_weights, and adjustments must have the same length")

    weights = baseline * np.clip(adjustments, ADJUSTMENT_LOW, ADJUSTMENT_HIGH)
    length = len(returns)
    rewards = np.empty(length, dtype=float)
    gross = 0.0
    turnover = 0.0
    risk = 0.0
    drawdown_sum = 0.0
    max_drawdown = 0.0
    equity = 1.0
    peak = 1.0
    curve = [1.0]
    previous_weight = baseline[0] if length else 0.0
    for index in range(length):
        weight = weights[index]
        period_return = weight * returns[index]
        weight_change = abs(weight - previous_weight)
        previous_weight = weight
        equity += period_return
        peak = max(peak, equity)
        drawdown = peak - equity

        gross += period_return
        turnover += weight_change
        risk += weight * weight
        drawdown_sum += drawdown
        max_drawdown = max(max_drawdown, drawdown)
        curve.append(equity)
        rewards[index] = (
            period_return
            - costs.cost_rate * weight_change
            - costs.risk_exposure_penalty * weight * weight
            - costs.drawdown_penalty * drawdown
        )
    return rewards, gross, turnover, risk, drawdown_sum, max_drawdown, curve


def per_step_rewards(returns, baseline_weights, adjustments, costs) -> np.ndarray:
    """Return the per-step reward array (used by the PPO training loop)."""
    return _simulate(returns, baseline_weights, adjustments, costs)[0]


def simulate(returns, baseline_weights, adjustments, costs) -> SizingResult:
    rewards, gross, turnover, risk, drawdown_sum, max_drawdown, curve = _simulate(
        returns, baseline_weights, adjustments, costs
    )
    turnover_cost = costs.cost_rate * turnover
    risk_penalty = costs.risk_exposure_penalty * risk
    drawdown_penalty = costs.drawdown_penalty * drawdown_sum
    total_reward = float(np.sum(rewards))
    return SizingResult(
        total_reward=total_reward,
        gross_return=float(gross),
        turnover_cost=float(turnover_cost),
        risk_penalty=float(risk_penalty),
        drawdown_penalty=float(drawdown_penalty),
        max_drawdown=float(max_drawdown),
        total_turnover=float(turnover),
        equity_curve=tuple(curve),
    )


def evaluate_policy(policy, returns, baseline_weights, costs, window: int = 20) -> SizingResult:
    """Run a sizing policy through the environment and report its reward."""
    states = build_states(returns, baseline_weights, window)
    adjustments = np.array([clamp_adjustment(policy(state)) for state in states])
    return simulate(returns, baseline_weights, adjustments, costs)


def identity_sizer(_state) -> float:
    """Fixed/equal-weight baseline: no adjustment to the baseline position."""
    return 1.0


def volatility_target_sizer(vol_target: float):
    """Inverse-volatility sizing, clamped to the [0.8, 1.2] band."""

    def sizer(state) -> float:
        volatility = float(state[2])
        if volatility <= 0:
            return 1.0
        return clamp_adjustment(vol_target / volatility)

    return sizer


def should_fallback(candidate: SizingResult, baseline: SizingResult) -> bool:
    """Fall back to the baseline when a candidate does not improve total reward."""
    return candidate.total_reward <= baseline.total_reward
