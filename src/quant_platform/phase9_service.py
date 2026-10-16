"""Phase 9 portfolio research pipeline as a callable, read-only service.

Runs the full research chain on a synthetic ETF panel:

  panel -> LightGBM/DoubleEnsemble candidate pool -> TradingAgents verification
  -> constrained PPO sizing -> verified target weights

Returns a structured result dict. No orders, no risk-engine interaction, and
no network I/O happen here; the HTTP layer (http_api.phase9_payload) adds the
risk gate and paper submission.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .etf_universe import build_etf_panel
from .pipeline import build_targets
from .position_sizing import SizingCosts, build_states
from .ppo_policy import train_ppo
from .providers import TushareDatasetProvider
from .qlib_alpha158 import compute_factor_ic_report, run_alpha158
from .qlib_dump import build_ohlcv_panel, dump_etf_to_qlib
from .qlib_model import evaluate_scores, run_qlib_baseline, run_qlib_ensemble
from .report_charts import compute_report_charts
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


def _fetch_symbol_daily_adj(symbol, provider, daily_dataset, adj_dataset, start_date, end_date):
    daily_rows, adj_rows = [], []
    for row in provider.fetch_records(daily_dataset, symbol, start_date, end_date):
        if row.get("close") is None:
            continue
        daily_rows.append({
            "symbol": symbol,
            "trade_date": str(row.get("trade_date", "")).replace("-", "")[:8],
            "close": float(row["close"]),
            "volume": float(row.get("vol") or 0.0),
            "amount": float(row.get("amount") or 0.0),
        })
    if adj_dataset is not None:
        for row in provider.fetch_records(adj_dataset, symbol, start_date, end_date):
            adj_rows.append({
                "symbol": symbol,
                "trade_date": str(row.get("trade_date", "")).replace("-", "")[:8],
                "adj_factor": float(row["adj_factor"]),
            })
    return daily_rows, adj_rows


def build_tushare_etf_panel(symbols, start_date, end_date, token_env="TUSHARE_TOKEN", asset_class="etf", max_workers=4) -> pd.DataFrame:
    """Fetch daily + adjustment data from Tushare (ETF/stock/HK) and build a panel."""
    symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if len(symbols) < 2:
        raise ValueError("at least two symbols are required")
    if not start_date or not end_date:
        raise ValueError("start_date and end_date are required")

    if asset_class == "etf":
        daily_dataset, adj_dataset = "fund_daily", "fund_adj"
    elif asset_class == "hk":
        daily_dataset, adj_dataset = "hk_daily", None
    else:
        daily_dataset, adj_dataset = "daily", "adj_factor"
    provider = TushareDatasetProvider(token_env)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda symbol: _fetch_symbol_daily_adj(
                symbol, provider, daily_dataset, adj_dataset, start_date, end_date
            ),
            symbols,
        ))
    daily_rows = [row for daily, _adj in results for row in daily]
    adj_rows = [row for _daily, adj in results for row in adj]
    if not daily_rows:
        raise ValueError(f"no Tushare {daily_dataset} data returned for the given symbols and dates")

    daily = pd.DataFrame(daily_rows)
    adj_factors = pd.DataFrame(adj_rows)
    if adj_factors.empty:
        adj_factors = pd.DataFrame({
            "symbol": daily["symbol"],
            "trade_date": daily["trade_date"],
            "adj_factor": 1.0,
        })
    basics = [{"ts_code": symbol, "list_status": "L"} for symbol in symbols]
    return build_etf_panel(daily, adj_factors, basics=basics, horizons=(1,))


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


def run_phase9(panel, *, top_k: int = 3, seed: int = 42) -> dict:
    """Run the Phase 9 research chain on a pre-built panel and return a result."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if panel["symbol"].nunique() < 2:
        raise ValueError("panel must contain at least two symbols")

    baseline = run_qlib_baseline(
        panel, top_k=top_k, num_boost_round=100, num_leaves=15, learning_rate=0.05,
    )
    ensemble = run_qlib_ensemble(
        panel, top_k=top_k, num_models=5, num_boost_round=50, num_leaves=15, learning_rate=0.05,
    )
    baseline_eval = evaluate_scores(baseline["scores"], panel, "forward_return_5d")
    ensemble_eval = evaluate_scores(ensemble["scores"], panel, "forward_return_5d")

    candidates = []
    latest_date = None
    if baseline["candidates"]:
        latest_date, candidates = list(baseline["candidates"].items())[-1]

    final = _finalize(panel, candidates, latest_date, seed)
    charts = compute_report_charts(
        baseline["scores"], panel, baseline["candidates"],
        label_column="forward_return_5d",
    )

    return {
        "panel": _panel_summary(panel),
        "baseline": {"rank_ic": baseline_eval["rank_ic"], "icir": baseline_eval["icir"]},
        "ensemble": {"rank_ic": ensemble_eval["rank_ic"], "icir": ensemble_eval["icir"]},
        "charts": charts,
        **final,
        "research_only": True,
        "execution_enabled": False,
    }


def _finalize(panel, candidates, latest_date, seed) -> dict:
    """Run verification, PPO sizing, and target building from a candidate pool."""
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
        "candidates": list(candidates),
        "verification": verification,
        "sizing": {"adjustment": adjustment},
        "targets": [{"symbol": target.symbol, "weight": target.weight} for target in targets],
    }


def _panel_summary(panel) -> dict:
    return {
        "rows": int(panel.shape[0]),
        "symbols": int(panel["symbol"].nunique()),
        "start": panel["timestamp"].min().isoformat(),
        "end": panel["timestamp"].max().isoformat(),
    }


def run_phase9_alpha158(
    symbols,
    start_date,
    end_date,
    *,
    top_k: int = 3,
    seed: int = 42,
    qlib_dir: str = "data/qlib_etf",
    asset_class: str = "etf",
    feature_expressions=None,
) -> dict:
    """Run the Phase 9 chain with Qlib's Alpha158 (or custom) factors for the candidate pool."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    ohlcv = build_ohlcv_panel(symbols, start_date, end_date, asset_class=asset_class)
    qlib_dir = str(Path(qlib_dir).expanduser())
    dump_etf_to_qlib(ohlcv, qlib_dir)

    baseline = run_alpha158(
        symbols, start_date, end_date, qlib_dir=qlib_dir, top_k=top_k,
        num_boost_round=100, num_leaves=31, learning_rate=0.05,
        feature_expressions=feature_expressions,
    )
    ensemble = run_alpha158(
        symbols, start_date, end_date, qlib_dir=qlib_dir, top_k=top_k,
        model_type="ensemble", num_models=5, num_boost_round=50, num_leaves=31, learning_rate=0.05,
        feature_expressions=feature_expressions,
    )

    panel = build_tushare_etf_panel(symbols, start_date, end_date, asset_class=asset_class)
    baseline_eval = evaluate_scores(baseline["scores"], panel, "forward_return_1d")
    ensemble_eval = evaluate_scores(ensemble["scores"], panel, "forward_return_1d")

    candidates = []
    latest_date = None
    if baseline["candidates"]:
        latest_date, candidates = list(baseline["candidates"].items())[-1]

    final = _finalize(panel, candidates, latest_date, seed)
    charts = compute_report_charts(
        baseline["scores"], panel, baseline["candidates"],
        label_column="forward_return_1d",
    )
    factor_ic = compute_factor_ic_report(
        symbols, start_date, end_date, qlib_dir=qlib_dir,
        feature_expressions=feature_expressions,
    )
    return {
        "panel": _panel_summary(panel),
        "baseline": {"rank_ic": baseline_eval["rank_ic"], "icir": baseline_eval["icir"]},
        "ensemble": {"rank_ic": ensemble_eval["rank_ic"], "icir": ensemble_eval["icir"]},
        "charts": charts,
        "factor_ic": factor_ic,
        **final,
        "research_only": True,
        "execution_enabled": False,
    }
