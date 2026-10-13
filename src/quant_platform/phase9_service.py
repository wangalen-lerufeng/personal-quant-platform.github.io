"""Phase 9 portfolio research pipeline as a callable, read-only service.

Runs the full research chain on a synthetic ETF panel:

  panel -> LightGBM/DoubleEnsemble candidate pool -> TradingAgents verification
  -> constrained PPO sizing -> verified target weights

Returns a structured result dict. No orders, no risk-engine interaction, and
no network I/O happen here; the HTTP layer (http_api.phase9_payload) adds the
risk gate and paper submission.
"""

import numpy as np
import pandas as pd

from .double_ensemble import DoubleEnsembleConfig, run_double_ensemble
from .etf_universe import build_etf_panel
from .ml_baseline import BaselineConfig, run_baseline
from .pipeline import build_targets
from .position_sizing import SizingCosts, build_states
from .ppo_policy import train_ppo
from .trading_agents import EvidenceItem, build_role_verdicts, synthesize_decision


def synthetic_etf_panel(symbols, n_days: int = 120) -> pd.DataFrame:
    """Build a deterministic synthetic ETF panel for demo/research purposes."""
    symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if len(symbols) < 2:
        raise ValueError("at least two symbols are required")
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    daily_rows, adj_rows = [], []
    for index, symbol in enumerate(symbols):
        base = 4.0 * (index + 1)
        drift = 0.002 * (index + 1)
        for day_index, timestamp in enumerate(dates):
            close = base * (1 + drift * day_index + 0.01 * np.sin(day_index / 5 + index))
            daily_rows.append({
                "symbol": symbol,
                "trade_date": timestamp.strftime("%Y%m%d"),
                "close": close,
                "volume": 1000 * (index + 1),
                "amount": 1_000_000 * (index + 1),
            })
            adj_rows.append({
                "symbol": symbol,
                "trade_date": timestamp.strftime("%Y%m%d"),
                "adj_factor": 1.0,
            })
    basics = [{"ts_code": symbol, "list_status": "L"} for symbol in symbols]
    return build_etf_panel(
        pd.DataFrame(daily_rows),
        pd.DataFrame(adj_rows),
        basics=basics,
        horizons=(5,),
    )


def _verification_evidence(panel, symbol, as_of, flag_volatile):
    symbol_panel = panel[panel["symbol"] == symbol].sort_values("timestamp")
    returns = symbol_panel["adj_close"].pct_change().dropna()
    volatility = float(returns.std())
    momentum = float(returns.tail(20).mean())
    technical_field = "high_volatility" if flag_volatile else "momentum"
    technical_note = "波动偏高" if flag_volatile else f"动量 {momentum:+.4f}"
    return [
        EvidenceItem("fundamental", "fund_basic", as_of, "net_profit", 1.0, "净利润为正"),
        EvidenceItem("technical", "daily", as_of, technical_field, volatility, technical_note),
        EvidenceItem("news", "anns_d", as_of, "announcement", "无重大利空", "无重大负面公告"),
        EvidenceItem("risk", "daily_basic", as_of, "amount", 1_000_000, "流动性充足"),
        EvidenceItem("bearish", "review", as_of, "no_objection", True, "反方未发现排除理由"),
    ]


def run_phase9(symbols, *, top_k: int = 3, seed: int = 42) -> dict:
    """Run the Phase 9 research chain and return a structured result."""
    symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if len(symbols) < 2:
        raise ValueError("at least two symbols are required")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    panel = synthetic_etf_panel(symbols)
    baseline = run_baseline(
        panel,
        config=BaselineConfig(n_estimators=100, num_leaves=15, min_child_samples=10, seed=seed),
        out_of_sample_start=None,
        top_k=top_k,
    )
    ensemble = run_double_ensemble(
        panel,
        config=DoubleEnsembleConfig(
            n_models=5, n_estimators=50, num_leaves=15, min_child_samples=10, seed=seed,
        ),
        out_of_sample_start=None,
        top_k=top_k,
    )

    candidates = []
    latest_date = None
    if baseline.candidates:
        latest_date, candidates = list(baseline.candidates.items())[-1]

    decisions = {}
    verification = []
    if candidates:
        volatilities = {
            symbol: float(panel[panel["symbol"] == symbol]["adj_close"].pct_change().std())
            for symbol in candidates
        }
        max_volatility_symbol = max(volatilities, key=volatilities.get)
        for symbol in candidates:
            evidence = _verification_evidence(
                panel, symbol, latest_date, flag_volatile=(symbol == max_volatility_symbol)
            )
            decision = synthesize_decision(symbol, build_role_verdicts(evidence))
            decisions[symbol] = decision
            verification.append({
                "symbol": symbol,
                "verdict": decision.verdict,
                "verified": decision.verified,
                "roles": {role.role: role.stance for role in decision.roles},
                "risk_tags": list(decision.risk_tags),
            })

    adjustment = 1.0
    if candidates:
        returns = (
            panel[panel["symbol"] == candidates[0]]
            .sort_values("timestamp")["adj_close"]
            .pct_change()
            .dropna()
            .to_numpy()
        )
        if len(returns) >= 20:
            policy = train_ppo(
                returns, np.ones_like(returns), costs=SizingCosts(), window=10,
                episodes=200, seed=seed, lr=0.2,
            )
            state = build_states(returns, np.ones_like(returns), 10)[-1]
            adjustment = float(np.clip(policy.predict(state), 0.8, 1.2))

    targets = build_targets(candidates, decisions, adjustment=adjustment)

    return {
        "panel": {
            "rows": int(panel.shape[0]),
            "symbols": int(panel["symbol"].nunique()),
            "start": panel["timestamp"].min().isoformat(),
            "end": panel["timestamp"].max().isoformat(),
        },
        "baseline": {
            "rank_ic": baseline.factor_report["horizons"]["forward_return_5d"]["rank_ic_mean"],
            "icir": baseline.factor_report["horizons"]["forward_return_5d"]["icir"],
        },
        "ensemble": {
            "rank_ic": ensemble.factor_report["horizons"]["forward_return_5d"]["rank_ic_mean"],
            "icir": ensemble.factor_report["horizons"]["forward_return_5d"]["icir"],
        },
        "candidates": list(candidates),
        "verification": verification,
        "sizing": {"adjustment": adjustment},
        "targets": [{"symbol": target.symbol, "weight": target.weight} for target in targets],
        "research_only": True,
        "execution_enabled": False,
    }
