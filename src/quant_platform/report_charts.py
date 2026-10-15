"""Chart-ready series for portfolio research results (read-only, no orders).

Turns model scores and a cross-sectional price panel into the equity/benchmark/
excess/drawdown/turnover/IC series the web console renders as charts. Pure
computation on in-memory DataFrames; no trading, no network I/O.
"""

import numpy as np
import pandas as pd


def compute_report_charts(
    scores: pd.DataFrame,
    panel: pd.DataFrame,
    candidates_by_date: dict,
    *,
    label_column: str = "forward_return_1d",
    annualization: int = 252,
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
    strategy_nav = np.cumprod(1.0 + strategy_rets_arr)
    benchmark_nav = np.cumprod(1.0 + benchmark_rets_arr)
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

    return {
        "dates": dates,
        "strategy_nav": [round(float(v), 6) for v in strategy_nav],
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
            "sample_days": n,
        },
    }
