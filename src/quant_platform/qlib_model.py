"""Qlib-backed candidate-pool models (LightGBM + DoubleEnsemble) for Phase 9.

Replaces the in-house lightgbm/double_ensemble modules with Qlib's reference
implementations (``qlib.contrib.model.gbdt.LGBModel`` and
``qlib.contrib.model.double_ensemble.DEnsembleModel``). A minimal dataset
wrapper feeds the same cross-sectional panel into Qlib's model interface.

Read-only research: only produces scores and candidate pools, never orders.
"""

import numpy as np
import pandas as pd

from .ml_baseline import DEFAULT_FEATURE_COLUMNS, compute_cross_sectional_features

_QLIB_INITIALIZED = False


def _ensure_qlib_init():
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED:
        return
    import qlib

    qlib.init(provider_uri="", region="cn")
    _QLIB_INITIALIZED = True


class _QlibDataset:
    """Minimal Qlib ``Dataset``-compatible wrapper over precomputed DataFrames."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
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


def _build_frames(panel, features, label_column, oos_boundary, valid_ratio=0.3):
    frame = compute_cross_sectional_features(panel)
    frame = frame.dropna(subset=[label_column, *features]).copy()
    boundary = pd.Timestamp(oos_boundary)
    timestamps = sorted(frame["timestamp"].unique())
    train_ts = [ts for ts in timestamps if ts < boundary]
    test_ts = [ts for ts in timestamps if ts >= boundary]
    split = max(1, int(len(train_ts) * (1 - valid_ratio)))
    valid_ts = train_ts[split:]
    train_ts = train_ts[:split]

    def build(ts_list):
        rows = frame[frame["timestamp"].isin(ts_list)].sort_values(["timestamp", "symbol"])
        index = pd.MultiIndex.from_arrays(
            [rows["timestamp"], rows["symbol"]], names=["datetime", "instrument"]
        )
        columns = pd.MultiIndex.from_tuples(
            [("feature", feature) for feature in features] + [("label", label_column)]
        )
        data = np.column_stack(
            [rows[feature].to_numpy() for feature in features] + [rows[label_column].to_numpy()]
        )
        return pd.DataFrame(data, index=index, columns=columns)

    return {"train": build(train_ts), "valid": build(valid_ts), "test": build(test_ts)}


def _scores_and_candidates(prediction: pd.Series, top_k: int) -> dict:
    scores = prediction.rename("score").reset_index()
    scores.columns = ["timestamp", "symbol", "score"]
    candidates = {}
    for timestamp, section in scores.groupby("timestamp"):
        top = section.sort_values("score", ascending=False)["symbol"].head(top_k)
        candidates[pd.Timestamp(timestamp).isoformat()] = [str(symbol) for symbol in top]
    return {"scores": scores, "candidates": candidates}


def run_qlib_baseline(
    panel,
    *,
    label_column: str = "forward_return_5d",
    feature_columns=DEFAULT_FEATURE_COLUMNS,
    oos_boundary=None,
    top_k: int = 10,
    **model_kwargs,
) -> dict:
    """Train Qlib's LGBModel and return OOS scores and a candidate pool."""
    from qlib.contrib.model.gbdt import LGBModel

    _ensure_qlib_init()
    boundary = _resolve_boundary(panel, label_column, oos_boundary)
    frames = _build_frames(panel, list(feature_columns), label_column, boundary)
    dataset = _QlibDataset(frames)
    model = LGBModel(**model_kwargs)
    model.fit(dataset, verbose_eval=-1)
    prediction = model.predict(dataset, segment="test")
    return _scores_and_candidates(prediction, top_k)


def run_qlib_ensemble(
    panel,
    *,
    label_column: str = "forward_return_5d",
    feature_columns=DEFAULT_FEATURE_COLUMNS,
    oos_boundary=None,
    top_k: int = 10,
    num_models: int = 6,
    **model_kwargs,
) -> dict:
    """Train Qlib's DoubleEnsemble (DEnsembleModel) and return OOS scores + candidates."""
    from qlib.contrib.model.double_ensemble import DEnsembleModel

    _ensure_qlib_init()
    boundary = _resolve_boundary(panel, label_column, oos_boundary)
    frames = _build_frames(panel, list(feature_columns), label_column, boundary)
    dataset = _QlibDataset(frames)
    model_kwargs.setdefault("decay", 0.5)
    model = DEnsembleModel(num_models=num_models, **model_kwargs)
    model.fit(dataset)
    prediction = model.predict(dataset, segment="test")
    return _scores_and_candidates(prediction, top_k)


def evaluate_scores(scores: pd.DataFrame, panel: pd.DataFrame, label_column: str) -> dict:
    """Evaluate model scores against forward returns via the factor harness."""
    from .factor_analysis import analyze_factor_panel

    merged = panel[["timestamp", "symbol", label_column]].merge(
        scores, on=["timestamp", "symbol"], how="inner"
    )
    report = analyze_factor_panel(
        merged,
        factor_column="score",
        return_columns=[label_column],
        date_column="timestamp",
        asset_column="symbol",
    )
    horizon = report["horizons"][label_column]
    return {"rank_ic": horizon["rank_ic_mean"], "icir": horizon["icir"]}


def _resolve_boundary(panel, label_column, oos_boundary):
    if oos_boundary is not None:
        return pd.Timestamp(oos_boundary)
    frame = compute_cross_sectional_features(panel).dropna(subset=[label_column])
    dates = sorted(frame["timestamp"].unique())
    if len(dates) < 3:
        raise ValueError("not enough timestamps for a train/test split")
    return dates[int(len(dates) * 0.7)]
