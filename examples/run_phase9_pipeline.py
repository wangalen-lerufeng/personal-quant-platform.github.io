"""Phase 9 end-to-end research pipeline demo (paper-only).

Runs the full chain on a synthetic ETF panel:

  data/features (step1) -> LightGBM + DoubleEnsemble candidate pool (step2/3)
  -> TradingAgents verification (step4) -> constrained PPO sizing (step5)
  -> deterministic risk gate -> paper simulation (step6)

Every output is read-only research (execution_enabled: false).
"""

import numpy as np
import pandas as pd

from quant_platform.double_ensemble import DoubleEnsembleConfig, run_double_ensemble
from quant_platform.etf_universe import build_etf_panel
from quant_platform.ml_baseline import BaselineConfig, run_baseline
from quant_platform.pipeline import build_targets, plan_orders, submit_to_simulation
from quant_platform.position_sizing import SizingCosts, build_states
from quant_platform.ppo_policy import train_ppo
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.simulation import PaperSimulationSession
from quant_platform.trading_agents import (
    EvidenceItem,
    build_role_verdicts,
    synthesize_decision,
)

SYMBOLS = ["510300.SH", "510500.SH", "159919.SZ", "512880.SH", "515790.SH"]


def build_synthetic_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=120)
    daily_rows, adj_rows = [], []
    for si, symbol in enumerate(SYMBOLS):
        base = 4.0 * (si + 1)
        drift = 0.002 * (si + 1)
        for di, ts in enumerate(dates):
            close = base * (1 + drift * di + 0.01 * np.sin(di / 5 + si))
            daily_rows.append({
                "symbol": symbol,
                "trade_date": ts.strftime("%Y%m%d"),
                "close": close,
                "volume": 1000 * (si + 1),
                "amount": 1_000_000 * (si + 1),
            })
            adj_rows.append({"symbol": symbol, "trade_date": ts.strftime("%Y%m%d"), "adj_factor": 1.0})
    basics = [{"ts_code": symbol, "list_status": "L"} for symbol in SYMBOLS]
    return build_etf_panel(
        pd.DataFrame(daily_rows),
        pd.DataFrame(adj_rows),
        basics=basics,
        horizons=(5,),
    )


def verification_evidence(panel: pd.DataFrame, symbol: str, as_of: str, *, flag_volatile: bool):
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


def main() -> None:
    panel = build_synthetic_panel()
    print(f"[step1] panel: {panel.shape[0]} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['timestamp'].min().date()} .. {panel['timestamp'].max().date()}")

    oos_start = pd.Timestamp("2026-04-01")
    baseline = run_baseline(
        panel,
        config=BaselineConfig(n_estimators=100, num_leaves=15, min_child_samples=10, seed=42),
        out_of_sample_start=oos_start,
        top_k=3,
    )
    ensemble = run_double_ensemble(
        panel,
        config=DoubleEnsembleConfig(
            n_models=5, n_estimators=50, num_leaves=15, min_child_samples=10, seed=42,
        ),
        out_of_sample_start=oos_start,
        top_k=3,
    )
    h_base = baseline.factor_report["horizons"]["forward_return_5d"]
    h_ens = ensemble.factor_report["horizons"]["forward_return_5d"]
    print(f"[step2] LightGBM   OOS rank_ic={h_base['rank_ic_mean']}  icir={h_base['icir']}")
    print(f"[step3] DoubleEnsemble OOS rank_ic={h_ens['rank_ic_mean']}  icir={h_ens['icir']}")

    latest_date, candidates = list(baseline.candidates.items())[-1]
    print(f"[step2/3] candidate pool @ {latest_date}: {candidates}")

    vols = {
        symbol: float(panel[panel["symbol"] == symbol]["adj_close"].pct_change().std())
        for symbol in candidates
    }
    max_vol_symbol = max(vols, key=vols.get)

    decisions = {}
    for symbol in candidates:
        evidence = verification_evidence(
            panel, symbol, latest_date, flag_volatile=(symbol == max_vol_symbol)
        )
        decisions[symbol] = synthesize_decision(symbol, build_role_verdicts(evidence))
        roles = ", ".join(f"{r.role}={r.stance}" for r in decisions[symbol].roles)
        print(f"[step4] {symbol}: verdict={decisions[symbol].verdict} "
              f"verified={decisions[symbol].verified} [{roles}]")

    returns = panel[panel["symbol"] == candidates[0]].sort_values("timestamp")["adj_close"].pct_change().dropna().to_numpy()
    policy = train_ppo(returns, np.ones_like(returns), costs=SizingCosts(), window=10, episodes=200, seed=0, lr=0.2)
    state = build_states(returns, np.ones_like(returns), 10)[-1]
    adjustment = float(np.clip(policy.predict(state), 0.8, 1.2))
    print(f"[step5] PPO adjustment for latest state {np.round(state, 3)}: {adjustment:.3f}")

    targets = build_targets(candidates, decisions, adjustment=adjustment)
    print(f"[step5] verified targets: {[(t.symbol, round(t.weight, 4)) for t in targets]}")

    risk_engine = RiskEngine(RiskLimits(max_order_quantity=1.0))
    plan = plan_orders({}, targets, risk_engine)
    print(f"[step6] risk-gated orders: {[{'symbol': o.symbol, 'side': o.side.value, 'qty': round(o.quantity, 4)} for o in plan.orders]}")
    print(f"[step6] rejected: {plan.rejected}")

    session = PaperSimulationSession("paper-main", initial_cash=1_000_000.0)
    submission = submit_to_simulation(plan, session)
    print(f"[step6] paper submission: {submission['submitted']}")
    print(f"[audit] {[(e.stage, e.symbol) for e in plan.audit]}")


if __name__ == "__main__":
    main()
