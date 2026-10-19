"""Chart-ready series for portfolio research results (read-only, no orders).

Turns model scores and a cross-sectional price panel into the equity/benchmark/
excess/drawdown/turnover/IC series the web console renders as charts. Pure
computation on in-memory DataFrames; no trading, no network I/O.
"""

import math

import numpy as np
import pandas as pd

from .cost_model import cost_breakdown


def compute_report_charts(
    scores: pd.DataFrame,
    panel: pd.DataFrame,
    candidates_by_date: dict,
    *,
    label_column: str = "forward_return_1d",
    annualization: int = 252,
    cost_rate: float = 0.003,
    cost_model: dict | None = None,
) -> dict | None:
    """Build chart series + scalar metrics for a model's out-of-sample backtest.

    ``scores`` has columns ``timestamp, symbol, score``; ``panel`` has
    ``timestamp, symbol, <label_column>``; ``candidates_by_date`` maps an ISO
    timestamp to the top-k symbols selected on that date.
    """
    if scores is None or panel is None or not candidates_by_date:
        return None
    if scores.empty or panel.empty or label_column not in panel.columns:
        return None

    score_frame = scores[["timestamp", "symbol", "score"]].copy()
    score_frame["timestamp"] = pd.to_datetime(score_frame["timestamp"])
    score_frame = score_frame.dropna(subset=["score"])

    panel_ret = panel[["timestamp", "symbol", label_column]].copy()
    panel_ret["timestamp"] = pd.to_datetime(panel_ret["timestamp"])
    panel_ret = panel_ret.dropna(subset=[label_column])
    if panel_ret.empty:
        return None

    merged = score_frame.merge(
        panel_ret[["timestamp", "symbol", label_column]],
        on=["timestamp", "symbol"],
        how="inner",
    )
    if merged.empty:
        return None

    dates: list[str] = []
    strategy_rets: list[float] = []
    benchmark_rets: list[float] = []
    turnovers: list[float] = []
    rank_ics: list[float | None] = []
    pearson_ics: list[float | None] = []

    prev_held: set[str] | None = None
    for key in sorted(candidates_by_date.keys()):
        ts = pd.Timestamp(key)
        held = [str(s) for s in candidates_by_date[key] if str(s)]
        if not held:
            continue

        held_rets = panel_ret[
            (panel_ret["timestamp"] == ts) & (panel_ret["symbol"].isin(held))
        ][label_column]
        bench_rets = panel_ret[panel_ret["timestamp"] == ts][label_column]
        if held_rets.empty or bench_rets.empty:
            continue

        held_set = set(held)
        if prev_held is None:
            turnover = 1.0  # initial entry
        else:
            turnover = len(held_set - prev_held) / max(1, len(held_set))
        prev_held = held_set

        day = merged[merged["timestamp"] == ts]
        rank_ic = pearson_ic = None
        if len(day) >= 3:
            rank_ic = float(day["score"].corr(day[label_column], method="spearman"))
            pearson_ic = float(day["score"].corr(day[label_column], method="pearson"))

        dates.append(ts.strftime("%Y-%m-%d"))
        strategy_rets.append(float(held_rets.mean()))
        benchmark_rets.append(float(bench_rets.mean()))
        turnovers.append(turnover)
        rank_ics.append(rank_ic)
        pearson_ics.append(pearson_ic)

    if not dates:
        return None

    strategy_rets_arr = np.asarray(strategy_rets, dtype=float)
    benchmark_rets_arr = np.asarray(benchmark_rets, dtype=float)
    turnovers_arr = np.asarray(turnovers, dtype=float)
    round_trip = cost_model["round_trip_rate"] if cost_model is not None else cost_rate
    strategy_nav = np.cumprod(1.0 + strategy_rets_arr)
    benchmark_nav = np.cumprod(1.0 + benchmark_rets_arr)
    net_rets = strategy_rets_arr - turnovers_arr * round_trip
    strategy_nav_net = np.cumprod(1.0 + net_rets)
    excess = strategy_nav - benchmark_nav
    daily_excess = strategy_rets_arr - benchmark_rets_arr

    running_peak = np.maximum.accumulate(strategy_nav)
    drawdown = strategy_nav / running_peak - 1.0

    n = len(daily_excess)
    ex_std = float(daily_excess.std(ddof=1)) if n > 1 else 0.0
    information_ratio = (
        float(daily_excess.mean()) / ex_std * np.sqrt(annualization) if ex_std > 0 else None
    )

    strategy_total = float(strategy_nav[-1] - 1.0)
    net_return_after_cost = float(strategy_nav_net[-1] - 1.0)
    strategy_ann = float(strategy_nav[-1] ** (annualization / max(1, n)) - 1.0) if n else 0.0
    strat_ann_vol = float(strategy_rets_arr.std(ddof=1) * np.sqrt(annualization)) if n > 1 else 0.0
    sharpe = strategy_ann / strat_ann_vol if strat_ann_vol > 0 else None

    benchmark_ann = float(benchmark_nav[-1] ** (annualization / max(1, n)) - 1.0) if n else 0.0
    bench_ann_vol = float(benchmark_rets_arr.std(ddof=1) * np.sqrt(annualization)) if n > 1 else 0.0
    beta = alpha = sortino = None
    if n > 1:
        cov = float(np.cov(strategy_rets_arr, benchmark_rets_arr)[0, 1])
        var_b = float(np.var(benchmark_rets_arr, ddof=1))
        if var_b > 0:
            beta = cov / var_b
            alpha = strategy_ann - beta * benchmark_ann
    downside = strategy_rets_arr[strategy_rets_arr < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(annualization)) if len(downside) > 1 else None
    if downside_dev and downside_dev > 0:
        sortino = strategy_ann / downside_dev
    daily_win_rate = float((strategy_rets_arr > 0).mean()) if n else None
    wins = strategy_rets_arr[strategy_rets_arr > 0]
    losses = strategy_rets_arr[strategy_rets_arr < 0]
    profit_loss_ratio = None
    if len(wins) and len(losses):
        avg_loss = float(abs(losses.mean()))
        if avg_loss > 0:
            profit_loss_ratio = float(wins.mean()) / avg_loss

    valid_ics = [ic for ic in rank_ics if ic is not None and not np.isnan(ic)]
    mean_ic = float(np.mean(valid_ics)) if valid_ics else None
    ic_std = float(np.std(valid_ics)) if len(valid_ics) > 1 else None
    ic_ir = mean_ic / ic_std if ic_std else None
    ic_positive_ratio = (
        float(sum(1 for v in valid_ics if v > 0) / len(valid_ics)) if valid_ics else None
    )
    ic_t_stat = ic_p_value = None
    if valid_ics and ic_std and ic_std > 0:
        n_ic = len(valid_ics)
        ic_t_stat = float(mean_ic / (ic_std / math.sqrt(n_ic)))
        ic_p_value = float(2 * (1 - 0.5 * (1 + math.erf(abs(ic_t_stat) / math.sqrt(2)))))
    rolling_ic_mean = (
        float(np.mean(valid_ics[-20:])) if len(valid_ics) >= 20 else None
    )

    cost_breakdown_report = None
    if cost_model is not None:
        total_turnover = float(np.sum(turnovers_arr))
        cost_breakdown_report = cost_breakdown(total_turnover, cost_model)
        cost_breakdown_report["min_commission"] = cost_model.get("min_commission")

    return {
        "dates": dates,
        "strategy_nav": [round(float(v), 6) for v in strategy_nav],
        "strategy_nav_net": [round(float(v), 6) for v in strategy_nav_net],
        "benchmark_nav": [round(float(v), 6) for v in benchmark_nav],
        "excess": [round(float(v), 6) for v in excess],
        "drawdown": [round(float(v), 6) for v in drawdown],
        "turnover": [round(float(v), 6) for v in turnovers],
        "ic_series": [None if v is None else round(float(v), 6) for v in rank_ics],
        "ic_pearson": [None if v is None else round(float(v), 6) for v in pearson_ics],
        "metrics": {
            "information_ratio": round(information_ratio, 4) if information_ratio is not None else None,
            "max_drawdown": round(float(drawdown.min()), 4),
            "strategy_cumulative_return": round(strategy_total, 4),
            "net_return_after_cost": round(net_return_after_cost, 4),
            "cost_rate": round(round_trip, 6),
            "cost_drag": round(strategy_total - net_return_after_cost, 4),
            "cost_label": cost_model.get("label") if cost_model is not None else None,
            "cost_breakdown": cost_breakdown_report,
            "benchmark_cumulative_return": round(float(benchmark_nav[-1] - 1.0), 4),
            "annualized_return": round(strategy_ann, 4),
            "annualized_volatility": round(strat_ann_vol, 4),
            "benchmark_volatility": round(bench_ann_vol, 4),
            "sharpe": round(sharpe, 4) if sharpe is not None else None,
            "alpha": round(alpha, 4) if alpha is not None else None,
            "beta": round(beta, 4) if beta is not None else None,
            "sortino": round(sortino, 4) if sortino is not None else None,
            "daily_win_rate": round(daily_win_rate, 4) if daily_win_rate is not None else None,
            "profit_loss_ratio": round(profit_loss_ratio, 4) if profit_loss_ratio is not None else None,
            "mean_turnover": round(float(np.mean(turnovers)), 4),
            "mean_ic": round(mean_ic, 4) if mean_ic is not None else None,
            "ic_ir": round(ic_ir, 4) if ic_ir is not None else None,
            "ic_positive_ratio": round(ic_positive_ratio, 4) if ic_positive_ratio is not None else None,
            "ic_t_stat": round(ic_t_stat, 4) if ic_t_stat is not None else None,
            "ic_p_value": round(ic_p_value, 4) if ic_p_value is not None else None,
            "rolling_ic_mean": round(rolling_ic_mean, 4) if rolling_ic_mean is not None else None,
            "sample_days": n,
        },
    }


def _classify_dimensions(
    trend: float, volatility: float, liquidity_ratio: float | None = None
) -> dict:
    """多维市场状态：趋势 / 波动 / 流动性 / 异常 四个维度，只用过去数据。

    ``trend`` 为窗口累计收益，``volatility`` 为窗口日收益标准差，``liquidity_ratio``
    为当日市场成交额 / 其滚动中位数。阈值为研究启发式，非拟合参数。
    """
    if volatility is None or volatility <= 0:
        return {"trend": "range", "volatility": "low", "liquidity": "normal", "anomaly": "data_issue"}
    if abs(trend) >= volatility:
        trend_dim = "up" if trend > 0 else "down"
    else:
        trend_dim = "range"
    if volatility >= 0.025:
        vol_dim = "high"
    elif volatility <= 0.008:
        vol_dim = "low"
    else:
        vol_dim = "mid"
    liq_dim = "tight" if (liquidity_ratio is not None and liquidity_ratio < 0.5) else "normal"
    return {"trend": trend_dim, "volatility": vol_dim, "liquidity": liq_dim, "anomaly": "none"}


def _state_label(dims: dict) -> str:
    trend_names = {"up": "趋势上涨", "down": "趋势下跌", "range": "震荡"}
    vol_names = {"high": "高波动", "mid": "中波动", "low": "低波动"}
    parts = [trend_names.get(dims.get("trend"), dims.get("trend"))]
    if dims.get("volatility") != "mid":
        parts.append(vol_names.get(dims.get("volatility"), dims.get("volatility")))
    if dims.get("liquidity") == "tight":
        parts.append("流动性紧张")
    if dims.get("anomaly") not in (None, "none"):
        parts.append("数据异常")
    return "·".join(parts)


def compute_market_state_report(
    scores: pd.DataFrame,
    panel: pd.DataFrame,
    label_column: str,
    candidates_by_date: dict,
    *,
    window: int = 20,
    liquidity_window: int = 20,
) -> list[dict] | None:
    """Group OOS dates by trailing market regime and report per-state diagnostics.

    The benchmark is the equal-weighted mean of symbol ``adj_close``. Each date is
    classified from its trailing ``window`` returns and, when ``amount`` is present,
    its trailing ``liquidity_window`` market amount (no future data). Rank IC,
    top-group forward return, turnover and max drawdown are reported per regime.
    """
    if not candidates_by_date or "adj_close" not in panel.columns:
        return None

    panel_c = panel.copy()
    panel_c["timestamp"] = pd.to_datetime(panel_c["timestamp"])
    daily_close = panel_c.groupby("timestamp")["adj_close"].mean().sort_index()
    bench_ret = daily_close.pct_change().dropna()
    if bench_ret.empty:
        return None

    daily_amount = None
    if "amount" in panel_c.columns:
        daily_amount = panel_c.groupby("timestamp")["amount"].sum().sort_index()

    score_frame = scores[["timestamp", "symbol", "score"]].copy()
    score_frame["timestamp"] = pd.to_datetime(score_frame["timestamp"])
    panel_ret = panel_c[["timestamp", "symbol", label_column]].dropna(subset=[label_column])
    merged = score_frame.merge(panel_ret, on=["timestamp", "symbol"], how="inner")

    rows: dict[pd.Timestamp, dict] = {}
    prev_held: set[str] | None = None
    for key in sorted(candidates_by_date.keys()):
        ts = pd.Timestamp(key)
        if ts not in bench_ret.index:
            continue
        position = bench_ret.index.get_loc(ts)
        dimensions = {
            "trend": "insufficient", "volatility": "insufficient",
            "liquidity": "insufficient", "anomaly": "none",
        }
        if position + 1 >= window:
            trailing = bench_ret.iloc[position + 1 - window : position + 1]
            liquidity_ratio = None
            if daily_amount is not None and ts in daily_amount.index:
                amount_pos = daily_amount.index.get_loc(ts)
                if amount_pos + 1 >= liquidity_window:
                    base = float(daily_amount.iloc[amount_pos + 1 - liquidity_window : amount_pos].median())
                    current = float(daily_amount.iloc[amount_pos])
                    if base > 0:
                        liquidity_ratio = current / base
            dimensions = _classify_dimensions(
                float((1.0 + trailing).prod() - 1.0),
                float(trailing.std()),
                liquidity_ratio,
            )
        state = _state_label(dimensions)

        held = [str(s) for s in candidates_by_date[key] if str(s)]
        if not held:
            continue
        held_set = set(held)
        turnover = 1.0 if prev_held is None else len(held_set - prev_held) / max(1, len(held_set))
        prev_held = held_set

        day = merged[merged["timestamp"] == ts]
        rank_ic = (
            float(day["score"].corr(day[label_column], method="spearman"))
            if len(day) >= 3
            else None
        )
        held_rets = panel_ret[
            (panel_ret["timestamp"] == ts) & (panel_ret["symbol"].isin(held))
        ][label_column]
        rows[ts] = {
            "state": state,
            "dimensions": dimensions,
            "rank_ic": rank_ic,
            "turnover": turnover,
            "top_return": float(held_rets.mean()) if not held_rets.empty else None,
        }

    if not rows:
        return None

    buckets: dict[str, dict[str, list[float]]] = {}
    dimensions_by_state: dict[str, dict] = {}
    for ts in sorted(rows.keys()):
        row = rows[ts]
        bucket = buckets.setdefault(row["state"], {"rank_ic": [], "turnover": [], "top_return": []})
        dimensions_by_state.setdefault(row["state"], row["dimensions"])
        for field in ("rank_ic", "turnover", "top_return"):
            if row[field] is not None:
                bucket[field].append(row[field])

    def _mean(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 6) if values else None

    def _max_drawdown(values: list[float]) -> float | None:
        if not values:
            return None
        nav = np.cumprod(1.0 + np.asarray(values, dtype=float))
        drawdown = nav / np.maximum.accumulate(nav) - 1.0
        return round(float(drawdown.min()), 6)

    report = []
    for state in sorted(buckets.keys()):
        bucket = buckets[state]
        report.append({
            "state": state,
            "dimensions": dimensions_by_state[state],
            "days": max(len(bucket["rank_ic"]), len(bucket["turnover"])),
            "rank_ic": _mean(bucket["rank_ic"]),
            "top_return": _mean(bucket["top_return"]),
            "turnover": _mean(bucket["turnover"]),
            "max_drawdown": _max_drawdown(bucket["top_return"]),
        })
    return report
