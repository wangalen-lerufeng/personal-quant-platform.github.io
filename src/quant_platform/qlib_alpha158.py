"""Qlib Alpha158 factor pipeline: real Qlib feature engine + models.

Uses Qlib's ``Alpha158`` handler (about 158 cross-sectional factors) by default,
or a custom ``DataHandlerLP`` with user-supplied expression factors. The data
must already be dumped to Qlib's ``.bin`` format via ``qlib_dump``.
"""

import pandas as pd

DEFAULT_LABEL = (["Ref($close, -1)/$close - 1"], ["LABEL0"])
FORWARD_RETURN_EXPR = "Ref($close, -1)/$close - 1"


def compute_factor_ic_report(
    symbols,
    start_date,
    end_date,
    *,
    qlib_dir: str,
    feature_expressions=None,
    top_n: int = 20,
    min_ic: float = 0.02,
    min_icir: float = 0.15,
    train_end=None,
    corr_threshold: float = 0.9,
) -> dict:
    """Return per-factor Rank IC + ICIR for the factor set used by the pipeline.

    For custom ``feature_expressions`` it reports exactly those factors; otherwise
    it reports the top-N of Alpha158's ~158 factors by absolute Rank IC. Each factor
    is also tagged ``keep``/``drop`` by the ``min_ic``/``min_icir`` thresholds so the
    console can suggest a trimmed factor list for iterative optimisation.

    When ``train_end`` is given, IC/ICIR are computed only over the training period
    ``[start_date, train_end]`` so factor selection does not peek into the test period.
    """
    import numpy as np
    from qlib.data import D

    from .qlib_dump import init_qlib

    init_qlib(qlib_dir)

    if feature_expressions:
        expressions = [str(e).strip() for e in feature_expressions if str(e).strip()]
        names = [f"F{i}" for i in range(len(expressions))]
    else:
        from qlib.contrib.data.loader import Alpha158DL

        expressions, names = Alpha158DL.get_feature_config()
    if not expressions:
        return {
            "factors": [], "total": 0, "kept": 0, "recommended": [],
            "min_ic": min_ic, "min_icir": min_icir, "train_end": train_end,
            "research_only": True, "execution_enabled": False,
        }

    ic_end = train_end or end_date
    frame = D.features(
        list(symbols), expressions + [FORWARD_RETURN_EXPR],
        start_time=start_date, end_time=ic_end,
    ).dropna(subset=[FORWARD_RETURN_EXPR])

    report = []
    for index, expr in enumerate(expressions):
        daily_ics = []
        for _, section in frame.groupby(level="datetime"):
            if len(section) < 3:
                continue
            ic = section[expr].corr(section[FORWARD_RETURN_EXPR], method="spearman")
            if ic is not None and not (isinstance(ic, float) and np.isnan(ic)):
                daily_ics.append(ic)
        if not daily_ics:
            continue
        daily_ics = np.asarray(daily_ics, dtype=float)
        mean_ic = float(daily_ics.mean())
        ic_std = float(daily_ics.std(ddof=1)) if len(daily_ics) > 1 else 0.0
        icir = round(mean_ic / ic_std, 4) if ic_std > 0 else None
        verdict = (
            "keep"
            if abs(mean_ic) >= min_ic and icir is not None and abs(icir) >= min_icir
            else "drop"
        )
        report.append({
            "name": names[index],
            "expression": expr,
            "rank_ic": round(mean_ic, 4),
            "icir": icir,
            "days": int(len(daily_ics)),
            "verdict": verdict,
        })

    report.sort(key=lambda item: abs(item["rank_ic"]), reverse=True)
    top_report = report[: max(1, int(top_n))]

    # 相关性聚类：在 Top-N 因子内，与已保留因子 |corr| >= 阈值的标记为冗余（剔除）
    if len(top_report) >= 2:
        exprs = [item["expression"] for item in top_report]
        corr = frame[exprs].corr()
        kept_indices = []
        for i, item in enumerate(top_report):
            if any(abs(corr.iloc[i, j]) >= corr_threshold for j in kept_indices):
                item["verdict"] = "drop"
                item["redundant"] = True
            else:
                kept_indices.append(i)

    recommended = [item["expression"] for item in report if item["verdict"] == "keep"]
    return {
        "factors": top_report,
        "total": len(report),
        "kept": len(recommended),
        "recommended": recommended,
        "min_ic": min_ic,
        "min_icir": min_icir,
        "train_end": train_end,
        "research_only": True,
        "execution_enabled": False,
    }


def analyze_single_symbol_factors(
    symbol,
    factor_expressions,
    start_date,
    end_date,
    *,
    qlib_dir: str = "data/qlib_factor",
    asset_class: str = "etf",
) -> dict:
    """Compute factor time series for a single symbol and return per-factor stats."""
    import numpy as np
    from qlib.data import D

    from .qlib_dump import build_ohlcv_panel, dump_etf_to_qlib, init_qlib

    symbol = str(symbol).strip()
    expressions = [str(expr).strip() for expr in factor_expressions if str(expr).strip()]
    if not symbol:
        raise ValueError("symbol is required")
    if not expressions:
        raise ValueError("at least one factor expression is required")

    ohlcv = build_ohlcv_panel([symbol], start_date, end_date, asset_class=asset_class)
    dump_etf_to_qlib(ohlcv, qlib_dir)
    init_qlib(qlib_dir)

    frame = D.features(
        [symbol], expressions + [FORWARD_RETURN_EXPR], start_time=start_date, end_time=end_date
    ).dropna()
    if frame.empty:
        raise ValueError(f"未取到 {symbol} 的因子数据，请检查代码/日期/权限")

    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in frame.index.get_level_values("datetime")]
    forward = frame[FORWARD_RETURN_EXPR].to_numpy()

    factors = []
    for index, expr in enumerate(expressions):
        values = frame[expr].to_numpy()
        mean = float(np.mean(values))
        std = float(np.std(values))
        latest = float(values[-1])
        percentile = float((values < latest).mean())
        ic = float(np.corrcoef(values, forward)[0, 1]) if len(values) > 2 else None
        factors.append({
            "name": f"F{index}",
            "expression": expr,
            "mean": mean,
            "std": std,
            "latest": latest,
            "percentile": percentile,
            "ic": ic,
            "series": [{"date": dates[i], "value": float(values[i])} for i in range(len(values))],
        })

    return {
        "symbol": symbol,
        "start": dates[0],
        "end": dates[-1],
        "factors": factors,
        "research_only": True,
        "execution_enabled": False,
    }


def _make_custom_handler(expressions, instruments, start_time, end_time):
    """Build a DataHandlerLP with user-supplied factor expressions."""
    from qlib.contrib.data.handler import DataHandlerLP, _DEFAULT_LEARN_PROCESSORS

    names = [f"F{index}" for index in range(len(expressions))]
    data_loader = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (list(expressions), names),
                "label": DEFAULT_LABEL,
            },
            "freq": "day",
        },
    }

    class _CustomHandler(DataHandlerLP):
        pass

    return _CustomHandler(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        data_loader=data_loader,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
    )


class _PrecomputedDataset:
    """Minimal Qlib Dataset wrapper over precomputed train/valid/test frames."""

    def __init__(self, frames):
        self._frames = frames
        self.segments = {name: None for name in frames}

    def prepare(self, segments, col_set="__all", data_key=None, **kwargs):
        if isinstance(segments, (list, tuple)):
            return tuple(self._one(segment, col_set) for segment in segments)
        return self._one(segments, col_set)

    def _one(self, segment, col_set):
        frame = self._frames[segment]
        if col_set == "__all" or col_set is None:
            return frame
        return frame[col_set]


def _build_dataset(frame, expressions, start, train_end, valid_end, end):
    """将 D.features 的结果切成 train/valid/test 并包装成 Qlib Dataset。"""
    frame = frame.copy()
    if "datetime" in frame.index.names and frame.index.names[0] != "datetime":
        frame = frame.reorder_levels(["datetime", "instrument"]).sort_index()

    def _map_col(col):
        return ("label", col) if col == FORWARD_RETURN_EXPR else ("feature", col)

    frame.columns = pd.MultiIndex.from_tuples([_map_col(col) for col in frame.columns])

    def seg(seg_start, seg_end):
        idx = frame.index.get_level_values("datetime")
        return frame[(idx >= seg_start) & (idx < seg_end)]

    return _PrecomputedDataset({
        "train": seg(start, train_end),
        "valid": seg(train_end, valid_end),
        "test": seg(valid_end, end),
    })


def run_alpha158(
    symbols,
    start_date,
    end_date,
    *,
    qlib_dir,
    top_k: int = 10,
    model_type: str = "lgb",
    num_models: int = 5,
    feature_expressions=None,
    preprocess: bool = True,
    **model_kwargs,
) -> dict:
    """Train a Qlib model on Alpha158 (or custom) factors and return scores + candidates.

    原始因子经 ``D.features`` 计算后，先过一遍因子预处理（MAD 去极值 + 截面标准化），
    再喂给 LightGBM/DoubleEnsemble。
    """
    from qlib.contrib.model.double_ensemble import DEnsembleModel
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data import D

    from .factor_preprocessing import preprocess_factor_panel
    from .qlib_dump import init_qlib

    init_qlib(qlib_dir)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = int((end - start).days)
    train_end = start + pd.Timedelta(days=int(total_days * 0.7))
    valid_end = start + pd.Timedelta(days=int(total_days * 0.85))

    if feature_expressions:
        expressions = [str(e).strip() for e in feature_expressions if str(e).strip()]
    else:
        from qlib.contrib.data.loader import Alpha158DL

        expressions, _names = Alpha158DL.get_feature_config()
    if not expressions:
        raise ValueError("no factor expressions")

    frame = D.features(
        list(symbols), expressions + [FORWARD_RETURN_EXPR],
        start_time=start_date, end_time=end_date,
    ).dropna(subset=[FORWARD_RETURN_EXPR])

    if preprocess:
        frame = preprocess_factor_panel(frame, expressions)

    dataset = _build_dataset(frame, expressions, start, train_end, valid_end, end)

    if model_type == "ensemble":
        model_kwargs.setdefault("decay", 0.5)
        model = DEnsembleModel(num_models=num_models, **model_kwargs)
    else:
        model = LGBModel(**model_kwargs)

    model.fit(dataset)
    prediction = model.predict(dataset, segment="test")
    scores = prediction.rename("score").reset_index()
    scores.columns = ["timestamp", "symbol", "score"]

    candidates = {}
    for timestamp, section in scores.groupby("timestamp"):
        top = section.sort_values("score", ascending=False)["symbol"].head(top_k)
        candidates[pd.Timestamp(timestamp).isoformat()] = [str(symbol) for symbol in top]

    return {"scores": scores, "candidates": candidates}
