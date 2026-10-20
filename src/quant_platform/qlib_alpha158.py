"""Qlib Alpha158 factor pipeline: real Qlib feature engine + models.

Uses Qlib's ``Alpha158`` handler (about 158 cross-sectional factors) by default,
or a custom ``DataHandlerLP`` with user-supplied expression factors. The data
must already be dumped to Qlib's ``.bin`` format via ``qlib_dump``.
"""

import numpy as np
import pandas as pd

DEFAULT_LABEL = (["Ref($close, -1)/$close - 1"], ["LABEL0"])
FORWARD_RETURN_EXPR = "Ref($close, -1)/$close - 1"


def forward_return_expression(horizon: int) -> str:
    if int(horizon) < 1:
        raise ValueError("label_horizon must be positive")
    return f"Ref($close, -{int(horizon)})/$close - 1"


def _effective_purge(label_horizon: int, purge: int, *, monthly_rebalance: bool) -> int:
    # Month-end samples are farther apart than a 5-day label window; shifting by
    # five sparse samples would incorrectly remove five months from each segment.
    return 0 if monthly_rebalance else max(int(purge), int(label_horizon))


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


def _build_dataset(frame, start, train_end, valid_start, valid_end, test_start, test_end, *, label_column=FORWARD_RETURN_EXPR):
    """将 D.features 的结果切成 train/valid/test 并包装成 Qlib Dataset。

    各段边界显式传入，便于在 train/valid/test 之间加入 purge/embargo 隔离区：
    train=[start, train_end)、valid=[valid_start, valid_end)、test=[test_start, test_end)。
    """
    frame = frame.copy()
    if "datetime" in frame.index.names and frame.index.names[0] != "datetime":
        frame = frame.reorder_levels(["datetime", "instrument"]).sort_index()

    def _map_col(col):
        return ("label", col) if col == label_column else ("feature", col)

    frame.columns = pd.MultiIndex.from_tuples([_map_col(col) for col in frame.columns])

    def seg(seg_start, seg_end):
        idx = frame.index.get_level_values("datetime")
        return frame[(idx >= seg_start) & (idx < seg_end)]

    return _PrecomputedDataset({
        "train": seg(start, train_end),
        "valid": seg(valid_start, valid_end),
        "test": seg(test_start, test_end),
    })


def _model_instance(model_type, num_models, seed, model_kwargs):
    from qlib.contrib.model.double_ensemble import DEnsembleModel
    from qlib.contrib.model.gbdt import LGBModel

    if model_type == "ensemble":
        kwargs = dict(model_kwargs)
        kwargs.setdefault("decay", 0.5)
        return DEnsembleModel(num_models=num_models, seed=seed, **kwargs)
    return LGBModel(seed=seed, **model_kwargs)


def _predict_scores(model, dataset):
    prediction = model.predict(dataset, segment="test")
    scores = prediction.rename("score").reset_index()
    scores.columns = ["timestamp", "symbol", "score"]
    return scores


def _segment_boundaries(frame, start, end, purge, embargo, walk_forward, n_walk_folds):
    """Return segment boundaries with purge/embargo between train/valid/test.

    Each fold is ``(train_end, valid_start, valid_end, test_start, test_end)`` (all
    trading-day timestamps). ``purge`` drops the last ``purge`` trading days of each
    segment whose labels would cross the boundary; ``embargo`` inserts ``embargo``
    trading days after each boundary before the next segment starts.
    """
    dates = pd.Index(frame.index.get_level_values("datetime")).unique().sort_values()
    n = len(dates)
    train_end_i = max(1, int(n * 0.7))
    valid_end_i = max(2, int(n * 0.85))

    def shift(idx, delta):
        return dates[min(n - 1, max(0, idx + delta))]

    if not walk_forward:
        return [(
            shift(train_end_i, -purge),
            shift(train_end_i, embargo),
            shift(valid_end_i, -purge),
            shift(valid_end_i, embargo),
            end,
        )]

    folds = []
    chunks = np.array_split(np.arange(train_end_i, n), n_walk_folds)
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        test_start_i = int(chunk[0])
        test_end = dates[min(n - 1, int(chunk[-1]))] + pd.Timedelta(days=1)
        valid_span = max(1, int(test_start_i * 0.15))
        fold_train_end_i = max(1, test_start_i - valid_span)
        folds.append((
            shift(fold_train_end_i, -purge),
            shift(fold_train_end_i, embargo),
            shift(test_start_i, -purge),
            shift(test_start_i, embargo),
            test_end,
        ))
    return folds


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
    industry_map: dict | None = None,
    log_mv=None,
    n_seeds: int = 1,
    seed: int = 42,
    walk_forward: bool = False,
    n_walk_folds: int = 3,
    select_factors: bool = True,
    min_ic: float = 0.0,
    min_icir: float = 0.0,
    corr_threshold: float = 0.9,
    purge: int = 1,
    embargo: int = 1,
    label_horizon: int = 1,
    eligible_index=None,
    rank_within_industry: bool = False,
    monthly_rebalance: bool = False,
    **model_kwargs,
) -> dict:
    """Train a Qlib model on Alpha158 (or custom) factors and return scores + candidates.

    原始因子经 ``D.features`` 计算后，先过一遍因子预处理（MAD 去极值 + 行业市值中性化
    + 截面标准化），再用训练期数据做因子筛选（IC 阈值 + 相关性聚类），仅保留因子进入
    LightGBM/DoubleEnsemble。``n_seeds > 1`` 时多种子取平均分数（``seed_scores`` 保留
    各种子分数）；``walk_forward=True`` 时用扩展窗口滚动训练；``purge``/``embargo`` 在
    train/valid/test 之间加入标签隔离区，防止跨界泄漏。
    """
    from qlib.data import D

    from .factor_preprocessing import preprocess_factor_panel
    from .factor_selection import select_factors as _select_factors
    from .qlib_dump import init_qlib

    init_qlib(qlib_dir)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    label_expression = forward_return_expression(label_horizon)

    if feature_expressions:
        expressions = [str(e).strip() for e in feature_expressions if str(e).strip()]
    else:
        from qlib.contrib.data.loader import Alpha158DL

        expressions, _names = Alpha158DL.get_feature_config()
    if not expressions:
        raise ValueError("no factor expressions")

    frame = D.features(
        list(symbols), expressions + [label_expression],
        start_time=start_date, end_time=end_date,
    ).dropna(subset=[label_expression])

    if eligible_index is not None:
        eligible = pd.MultiIndex.from_tuples(
            [(str(symbol), pd.Timestamp(day)) for symbol, day in eligible_index],
            names=["instrument", "datetime"],
        )
        normalized = frame.index
        if normalized.names != ["instrument", "datetime"]:
            normalized = normalized.reorder_levels(["instrument", "datetime"])
        frame = frame[normalized.isin(eligible)]
        if frame.empty:
            raise ValueError("dynamic universe produced no eligible model samples")

    if preprocess:
        frame = preprocess_factor_panel(
            frame, expressions,
            industry_map=industry_map, log_mv=log_mv,
        )

    # 因子筛选必须发生在模型训练之前，且只读取训练期数据
    selection = None
    train_exprs = expressions
    if select_factors:
        train_end = start + pd.Timedelta(days=int((end - start).days * 0.7))
        selection = _select_factors(
            frame, expressions, start=start, train_end=train_end,
            forward_column=label_expression,
            min_ic=min_ic, min_icir=min_icir, corr_threshold=corr_threshold,
        )
        train_exprs = selection["selected"]
        if not train_exprs:
            raise ValueError("因子筛选后无保留因子")

    train_frame = frame[[*train_exprs, label_expression]]
    # Keep human-readable Qlib expressions in reports, but use JSON-safe names at the model boundary.
    train_frame = train_frame.rename(
        columns={expr: f"F{index}" for index, expr in enumerate(train_exprs)}
    )
    folds = _segment_boundaries(
        frame, start, end,
        _effective_purge(label_horizon, purge, monthly_rebalance=monthly_rebalance),
        embargo, walk_forward, n_walk_folds,
    )

    fold_scores = []
    for fold_train_end, valid_start, valid_end, test_start, test_end in folds:
        dataset = _build_dataset(
            train_frame, start, fold_train_end, valid_start, valid_end, test_start, test_end,
            label_column=label_expression,
        )
        seed_scores = []
        for seed_offset in range(n_seeds):
            trial_seed = seed + seed_offset
            model = _model_instance(model_type, num_models, trial_seed, model_kwargs)
            state = np.random.get_state()
            np.random.seed(trial_seed)
            try:
                model.fit(dataset)
            finally:
                np.random.set_state(state)
            seed_scores.append(_predict_scores(model, dataset))
        fold_scores.append(seed_scores)

    seed_score_frames = [
        pd.concat([fold[seed_i] for fold in fold_scores], ignore_index=True)
        for seed_i in range(n_seeds)
    ]

    if n_seeds > 1:
        combined = pd.concat(seed_score_frames, ignore_index=True)
        scores = combined.groupby(["timestamp", "symbol"], as_index=False)["score"].mean()
    else:
        scores = seed_score_frames[0]

    if rank_within_industry:
        if not industry_map:
            raise ValueError("industry_map is required for industry ranking")
        from .research_universe import rank_scores_within_industry
        scores = rank_scores_within_industry(scores, industry_map)

    candidates = {}
    for timestamp, section in scores.groupby("timestamp"):
        top = section.sort_values("score", ascending=False)["symbol"].head(top_k)
        candidates[pd.Timestamp(timestamp).isoformat()] = [str(symbol) for symbol in top]

    return {
        "scores": scores,
        "candidates": candidates,
        "seed_scores": seed_score_frames,
        "factor_selection": selection,
        "label_horizon": int(label_horizon),
        "label_expression": label_expression,
    }
