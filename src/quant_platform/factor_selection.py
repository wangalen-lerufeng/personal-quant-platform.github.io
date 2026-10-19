"""因子筛选：训练期 IC 诊断 + 相关性聚类，输出最终因子清单。

只在训练期数据上选择因子，验证期/测试期不参与筛选。筛选结果直接传入
LightGBM/DoubleEnsemble，避免「事后报告」式的无效筛选。
"""

import numpy as np
import pandas as pd


def select_factors(
    frame: pd.DataFrame,
    expressions,
    *,
    start,
    train_end,
    forward_column,
    min_ic: float = 0.02,
    min_icir: float = 0.15,
    corr_threshold: float = 0.9,
    top_n: int = 20,
) -> dict:
    """从训练期样本中选择最终因子清单。

    ``frame`` 是预处理后的 Qlib 特征面板（MultiIndex ``datetime x instrument``），
    包含 ``expressions`` 与 ``forward_column``。因子先用 |Rank IC| / |ICIR| 阈值在
    ``[start, train_end)`` 过滤，再按 |Spearman 相关| 聚类去冗余。
    """
    idx = frame.index.get_level_values("datetime")
    train = frame[(idx >= start) & (idx < train_end)].dropna(subset=[forward_column])

    report = []
    for index, expr in enumerate(expressions):
        daily_ics = []
        for _, section in train.groupby(level="datetime"):
            if len(section) < 3:
                continue
            ic = section[expr].corr(section[forward_column], method="spearman")
            if ic is not None and not (isinstance(ic, float) and np.isnan(ic)):
                daily_ics.append(ic)
        if not daily_ics:
            report.append({
                "name": f"F{index}", "expression": expr,
                "rank_ic": None, "icir": None, "days": 0, "verdict": "drop",
            })
            continue
        daily_ics = np.asarray(daily_ics, dtype=float)
        mean_ic = float(daily_ics.mean())
        ic_std = float(daily_ics.std(ddof=1)) if len(daily_ics) > 1 else 0.0
        icir = mean_ic / ic_std if ic_std > 0 else None
        verdict = (
            "keep"
            if abs(mean_ic) >= min_ic and icir is not None and abs(icir) >= min_icir
            else "drop"
        )
        report.append({
            "name": f"F{index}", "expression": expr,
            "rank_ic": round(mean_ic, 4),
            "icir": round(icir, 4) if icir is not None else None,
            "days": int(len(daily_ics)), "verdict": verdict,
        })

    kept = [item for item in report if item["verdict"] == "keep"]
    kept.sort(key=lambda item: abs(item["rank_ic"] or 0.0), reverse=True)
    if len(kept) >= 2 and corr_threshold < 1.0:
        exprs = [item["expression"] for item in kept]
        corr = train[exprs].corr()
        kept_indices = []
        for i, item in enumerate(kept):
            if any(abs(corr.iloc[i, j]) >= corr_threshold for j in kept_indices):
                item["verdict"] = "drop"
                item["redundant"] = True
            else:
                kept_indices.append(i)

    recommended = [item["expression"] for item in report if item["verdict"] == "keep"]
    if not recommended:
        # 阈值过严时回退到 |Rank IC| 最高的单因子，保证模型有输入，交由准入闸门裁决。
        ranked = sorted(
            (item for item in report if item["rank_ic"] is not None),
            key=lambda item: abs(item["rank_ic"]), reverse=True,
        )
        if ranked:
            ranked[0]["verdict"] = "keep"
            recommended = [ranked[0]["expression"]]

    report.sort(key=lambda item: abs(item["rank_ic"] or 0.0), reverse=True)
    top_report = report[: max(1, int(top_n))]

    return {
        "selected": recommended,
        "raw_count": len(expressions),
        "selected_count": len(recommended),
        "factors": top_report,
        "total": len(report),
        "kept": len(recommended),
        "recommended": recommended,
        "min_ic": min_ic,
        "min_icir": min_icir,
        "train_end": pd.Timestamp(train_end).isoformat(),
    }
