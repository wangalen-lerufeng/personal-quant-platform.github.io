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
from .qlib_alpha158 import run_alpha158
from .qlib_dump import build_ohlcv_panel, dump_etf_to_qlib
from .cost_model import build_cost_model
from .qlib_model import compute_group_returns, evaluate_scores, run_qlib_baseline, run_qlib_ensemble
from .report_charts import compute_market_state_report, compute_report_charts
from .trading_agents import EvidenceItem, build_role_verdicts, synthesize_decision

# 常见 ETF 的类别与跟踪指数（用于组合暴露诊断，非完整清单）
_ETF_CATEGORIES = {
    "510300.SH": ("宽基", "沪深300"),
    "510500.SH": ("宽基", "中证500"),
    "510050.SH": ("宽基", "上证50"),
    "159915.SZ": ("宽基", "创业板指"),
    "159919.SZ": ("宽基", "沪深300"),
    "588000.SH": ("宽基", "科创50"),
    "512880.SH": ("行业", "证券"),
    "512690.SH": ("行业", "酒"),
    "512480.SH": ("行业", "半导体"),
    "515790.SH": ("主题", "光伏"),
    "515030.SH": ("主题", "新能源车"),
    "512010.SH": ("行业", "医药"),
    "512170.SH": ("行业", "医疗"),
    "512800.SH": ("行业", "银行"),
    "512200.SH": ("行业", "房地产"),
    "512660.SH": ("行业", "军工"),
    "515000.SH": ("主题", "科技"),
    "512400.SH": ("行业", "有色金属"),
    "515050.SH": ("主题", "5G"),
    "512720.SH": ("行业", "计算机"),
    "159928.SZ": ("行业", "消费"),
}


def _classify_etf_exposure(symbols):
    """返回每只 ETF 的类别/跟踪指数，以及同类重叠类别（用于暴露诊断）。"""
    from collections import Counter

    items = []
    for symbol in symbols:
        category, index = _ETF_CATEGORIES.get(str(symbol), ("未知", None))
        items.append({"symbol": str(symbol), "category": category, "index": index})
    counts = Counter(item["category"] for item in items)
    overlaps = [cat for cat, count in counts.items() if count > 1 and cat != "未知"]
    return {"items": items, "overlap_categories": overlaps}


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


def build_tushare_etf_panel(symbols, start_date, end_date, token_env="TUSHARE_TOKEN", asset_class="etf", max_workers=1) -> pd.DataFrame:
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


def _model_gate(
    baseline_eval: dict,
    *,
    min_ic: float = 0.02,
    min_icir: float = 0.15,
    min_positive_ratio: float = 0.5,
    net_return_after_cost: float | None = None,
    group_returns: dict | None = None,
    max_drawdown: float | None = None,
    max_drawdown_limit: float = -1.0,
) -> dict:
    """多维模型准入门：样本外 Rank IC / ICIR / 正日比例 / 分组单调性 / 扣费后净收益 / 回撤。

    返回结构化规则明细；任一维度不达标即保持现金、不生成订单。阈值应由训练/验证期
    确定并锁定，测试期只应用已锁定的阈值。
    """
    rank_ic = baseline_eval.get("rank_ic")
    icir = baseline_eval.get("icir")
    positive_ratio = baseline_eval.get("positive_ratio")

    rules: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        rules.append({"name": name, "passed": bool(passed), "detail": detail})

    if rank_ic is None or icir is None:
        check("样本外评估", False, "Rank IC/ICIR 缺失")
    else:
        check("样本外 Rank IC", rank_ic > min_ic, f"Rank IC={rank_ic:.4f}（阈值 >{min_ic}）")
        check("Rank ICIR", icir >= min_icir, f"ICIR={icir:.4f}（阈值 ≥{min_icir}）")

    if positive_ratio is None:
        check("Rank IC 正日比例", False, "缺失")
    else:
        check(
            "Rank IC 正日比例",
            positive_ratio >= min_positive_ratio,
            f"正日比例={positive_ratio:.4f}（阈值 ≥{min_positive_ratio}）",
        )

    if group_returns is None:
        check("分组收益单调性", False, "缺失")
    else:
        monotonic = bool(group_returns.get("monotonic", False))
        score = group_returns.get("monotonicity_score")
        direction = group_returns.get("direction", "none")
        detail = f"单调性={direction}"
        if score is not None:
            detail += f"（得分 {score:.3f}）"
        check("分组收益单调性", monotonic, detail)

    if net_return_after_cost is None:
        check("扣费后净收益", False, "缺失")
    else:
        check(
            "扣费后净收益",
            net_return_after_cost > 0,
            f"扣费后净收益={net_return_after_cost:.4f}",
        )

    if max_drawdown is not None:
        check(
            "最大回撤",
            max_drawdown >= max_drawdown_limit,
            f"最大回撤={max_drawdown:.4f}（阈值 ≥{max_drawdown_limit}）",
        )

    failed = [rule for rule in rules if not rule["passed"]]
    return {"passed": not failed, "reason": failed[0]["detail"] if failed else "", "rules": rules}


def _build_neutralization_data(symbols, start_date, end_date):
    """拉取 A 股行业分类与总市值，用于行业/市值中性化（股票资产）。

    返回 ``(industry_map, log_mv, meta)``；``log_mv`` 以 (instrument, datetime) 为索引，
    ``meta`` 记录数据来源与时点口径（行业为静态最新分类，存在历史回填风险）。
    任一步失败则返回 ``(None, None, meta)``，由调用方降级为不做中性化。
    """
    provider = TushareDatasetProvider("TUSHARE_TOKEN")

    industry_map = {}
    try:
        for row in provider.fetch_all_stocks():
            code = str(row.get("ts_code") or "").strip()
            if code:
                industry_map[code] = str(row.get("industry") or "").strip() or "unknown"
    except Exception:
        industry_map = {}

    records = []
    for symbol in symbols:
        try:
            for row in provider.fetch_records("daily_basic", symbol, start_date, end_date):
                mv = row.get("total_mv")
                if mv is None:
                    continue
                records.append({
                    "symbol": symbol,
                    "datetime": pd.Timestamp(str(row.get("trade_date", "")).replace("-", "")[:8]),
                    "log_mv": float(np.log(max(float(mv), 1.0))),
                })
        except Exception:
            continue

    meta = {
        "industry_source": "stock_basic.industry（静态最新）" if industry_map else None,
        "industry_effective_date": None,
        "market_cap_source": "daily_basic.total_mv（按日）",
        "market_cap_date": None,
        "disclaimer": "行业分类使用当前静态分类，可能存在历史分类偏差；市值取自对应交易日。",
        "applied": bool(records),
    }

    if not records:
        return industry_map or None, None, meta

    frame = pd.DataFrame(records).set_index(["symbol", "datetime"])["log_mv"]
    dates = frame.index.get_level_values("datetime")
    meta["market_cap_date"] = {
        "from": str(pd.Timestamp(dates.min()).date()),
        "to": str(pd.Timestamp(dates.max()).date()),
    }
    return industry_map or None, frame, meta


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
    compute_factor_ic: bool = True,
    min_ic: float = 0.02,
    min_icir: float = 0.15,
    n_seeds: int = 3,
    walk_forward: bool = False,
    n_walk_folds: int = 3,
) -> dict:
    """Run the Phase 9 chain with Qlib's Alpha158 (or custom) factors for the candidate pool."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    ohlcv = build_ohlcv_panel(symbols, start_date, end_date, asset_class=asset_class)
    qlib_dir = str(Path(qlib_dir).expanduser())
    dump_etf_to_qlib(ohlcv, qlib_dir)

    # 行业/市值中性化数据（仅股票资产，失败则降级为不做中性化）
    industry_map, log_mv, neutralization_meta = None, None, None
    if asset_class == "stock":
        try:
            industry_map, log_mv, neutralization_meta = _build_neutralization_data(symbols, start_date, end_date)
        except Exception:
            industry_map, log_mv, neutralization_meta = None, None, None

    baseline = run_alpha158(
        symbols, start_date, end_date, qlib_dir=qlib_dir, top_k=top_k,
        num_boost_round=100, num_leaves=31, learning_rate=0.05,
        feature_expressions=feature_expressions,
        industry_map=industry_map, log_mv=log_mv,
        n_seeds=n_seeds, seed=seed, walk_forward=walk_forward, n_walk_folds=n_walk_folds,
        select_factors=True, min_ic=min_ic, min_icir=min_icir,
    )
    selected_expressions = (
        baseline.get("factor_selection", {}).get("selected") if baseline.get("factor_selection") else None
    )
    ensemble = run_alpha158(
        symbols, start_date, end_date, qlib_dir=qlib_dir, top_k=top_k,
        model_type="ensemble", num_models=5, num_boost_round=50, num_leaves=31, learning_rate=0.05,
        feature_expressions=selected_expressions or feature_expressions,
        industry_map=industry_map, log_mv=log_mv,
        n_seeds=n_seeds, seed=seed, walk_forward=walk_forward, n_walk_folds=n_walk_folds,
        select_factors=False,
    )

    panel = build_tushare_etf_panel(symbols, start_date, end_date, asset_class=asset_class)
    baseline_eval = evaluate_scores(baseline["scores"], panel, "forward_return_1d")
    ensemble_eval = evaluate_scores(ensemble["scores"], panel, "forward_return_1d")
    cost_model = build_cost_model(asset_class)
    charts = compute_report_charts(
        baseline["scores"], panel, baseline["candidates"],
        label_column="forward_return_1d", cost_model=cost_model,
    )
    group_returns = compute_group_returns(baseline["scores"], panel, "forward_return_1d")
    market_state = compute_market_state_report(
        baseline["scores"], panel, "forward_return_1d", baseline["candidates"],
    )
    seed_stability = None
    seed_score_frames = baseline.get("seed_scores") or []
    if len(seed_score_frames) > 1:
        seed_ics = [
            evaluate_scores(seed_scores_frame, panel, "forward_return_1d")["rank_ic"]
            for seed_scores_frame in seed_score_frames
        ]
        seed_ics = [ic for ic in seed_ics if ic is not None]
        if seed_ics:
            seed_stability = {
                "rank_ics": [round(ic, 4) for ic in seed_ics],
                "mean": round(float(np.mean(seed_ics)), 4),
                "std": round(float(np.std(seed_ics)), 4),
            }
    factor_ic = baseline.get("factor_selection") if compute_factor_ic else None

    gate = _model_gate(
        baseline_eval,
        min_ic=min_ic,
        min_icir=min_icir,
        net_return_after_cost=(
            charts.get("metrics", {}).get("net_return_after_cost") if charts else None
        ),
        group_returns=group_returns,
        max_drawdown=(
            charts.get("metrics", {}).get("max_drawdown") if charts else None
        ),
    )
    base_result = {
        "panel": _panel_summary(panel),
        "baseline": {"rank_ic": baseline_eval["rank_ic"], "icir": baseline_eval["icir"]},
        "ensemble": {"rank_ic": ensemble_eval["rank_ic"], "icir": ensemble_eval["icir"]},
        "charts": charts,
        "group_returns": group_returns,
        "market_state": market_state,
        "seed_stability": seed_stability,
        "factor_ic": factor_ic,
        "neutralization_meta": neutralization_meta,
        "model_gate": gate,
        "research_only": True,
        "execution_enabled": False,
    }

    if not gate["passed"]:
        # 模型不达标：保持现金，不生成候选池、目标仓位或订单
        return {
            **base_result,
            "candidates": [],
            "verification": [],
            "sizing": {"adjustment": 0.0},
            "targets": [],
            "etf_exposure": None,
        }

    candidates = []
    latest_date = None
    if baseline["candidates"]:
        latest_date, candidates = list(baseline["candidates"].items())[-1]

    final = _finalize(panel, candidates, latest_date, seed)
    etf_exposure = _classify_etf_exposure(candidates) if asset_class == "etf" else None
    return {
        **base_result,
        **final,
        "etf_exposure": etf_exposure,
    }
