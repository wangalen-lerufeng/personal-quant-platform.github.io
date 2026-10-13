"""LightGBM cross-sectional baseline for the Phase 9 research pipeline.

Trains a reproducible cross-sectional model on causally-computed features to
rank ETF candidates, then evaluates the ranked scores through the existing
`factor_analysis.analyze_factor_panel` IC/Rank-IC harness. Read-only research:
no orders, no risk-engine interaction, and no future data in the features.

Uses the native LightGBM API (``lgb.Dataset`` / ``lgb.train`` / ``Booster``) so
the baseline does not require scikit-learn.
"""

import hashlib
from dataclasses import dataclass

import pandas as pd

from .factor_analysis import analyze_factor_panel

DEFAULT_WINDOWS = (20,)
DEFAULT_FEATURE_COLUMNS = (
    "return_1d",
    "momentum_20d",
    "volatility_20d",
    "ma_distance_20d",
    "volume_ratio_20d",
)


def compute_cross_sectional_features(
    panel: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Per-symbol, causally compute return/momentum/volatility/MA/volume features.

    ``panel`` needs ``symbol``, ``timestamp``, ``adj_close``, ``volume`` columns.
    Every feature uses only past or current bars (no future data).
    """
    required = {"symbol", "timestamp", "adj_close", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {', '.join(missing)}")
    if any(window < 1 for window in windows):
        raise ValueError("windows must be positive")

    frames = []
    for symbol, group in panel.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        part = group.sort_values("timestamp").copy()
        part["return_1d"] = part["adj_close"].pct_change()
        for window in windows:
            part[f"momentum_{window}d"] = part["adj_close"].pct_change(window)
            part[f"volatility_{window}d"] = part["return_1d"].rolling(window).std()
            part[f"ma_distance_{window}d"] = (
                part["adj_close"] / part["adj_close"].rolling(window).mean() - 1
            )
            part[f"volume_ratio_{window}d"] = (
                part["volume"] / part["volume"].rolling(window).mean()
            )
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class BaselineConfig:
    label_column: str = "forward_return_5d"
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 20
    seed: int = 42
    num_threads: int = 1


@dataclass(frozen=True)
class BaselineResult:
    scores: pd.DataFrame
    candidates: dict
    factor_report: dict
    metadata: dict

    def to_dict(self) -> dict:
        scores = self.scores.copy()
        scores["timestamp"] = scores["timestamp"].astype(str)
        return {
            "scores": scores.to_dict(orient="records"),
            "candidates": self.candidates,
            "factor_report": self.factor_report,
            "metadata": self.metadata,
            "research_only": True,
            "execution_enabled": False,
        }


def _import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError as error:
        raise ImportError(
            "lightgbm is required for the ML baseline; install it with "
            "`python3 -m pip install lightgbm`"
        ) from error


def _top_candidates(frame: pd.DataFrame, top_k: int) -> dict:
    candidates = {}
    for timestamp, section in frame.groupby("timestamp"):
        top = section.sort_values("score", ascending=False)["symbol"].head(top_k)
        candidates[pd.Timestamp(timestamp).isoformat()] = [str(symbol) for symbol in top]
    return candidates


def _metadata(config: BaselineConfig, panel: pd.DataFrame, model, oos_boundary, top_k: int) -> dict:
    key = panel[["timestamp", "symbol", "adj_close", "volume", "amount"]].sort_values(
        ["timestamp", "symbol"]
    )
    data_fingerprint = hashlib.sha256(key.to_csv(index=False).encode("utf-8")).hexdigest()
    model_signature = hashlib.sha256(model.model_to_string().encode("utf-8")).hexdigest()
    return {
        "label_column": config.label_column,
        "feature_columns": list(config.feature_columns),
        "hyperparameters": {
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "n_estimators": config.n_estimators,
            "min_child_samples": config.min_child_samples,
        },
        "seed": config.seed,
        "num_threads": config.num_threads,
        "top_k": top_k,
        "train_until": oos_boundary.isoformat(),
        "data_fingerprint": data_fingerprint,
        "model_signature": model_signature,
        "num_trees": int(model.num_trees()),
        "feature_importances": {
            str(feature): float(value)
            for feature, value in zip(config.feature_columns, model.feature_importance("gain"))
        },
        "research_only": True,
        "execution_enabled": False,
    }


def run_baseline(
    panel: pd.DataFrame,
    *,
    config: BaselineConfig | None = None,
    out_of_sample_start=None,
    top_k: int = 10,
) -> BaselineResult:
    """Train a LightGBM cross-sectional regressor and evaluate it out-of-sample."""
    config = config or BaselineConfig()
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    lgb = _import_lightgbm()

    frame = compute_cross_sectional_features(panel)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if config.label_column not in frame.columns:
        raise ValueError(f"panel is missing label column: {config.label_column}")
    frame = frame.dropna(subset=[config.label_column]).copy()

    if out_of_sample_start is None:
        dates = sorted(frame["timestamp"].unique())
        split_index = int(len(dates) * 0.7)
        if split_index >= len(dates):
            raise ValueError("not enough timestamps for a train/test split")
        out_of_sample_start = dates[split_index]
    oos_boundary = pd.Timestamp(out_of_sample_start)

    train = frame[frame["timestamp"] < oos_boundary].dropna(subset=list(config.feature_columns))
    test = frame[frame["timestamp"] >= oos_boundary].dropna(subset=list(config.feature_columns))
    if train.empty or test.empty:
        raise ValueError("out_of_sample_start must leave non-empty train and test periods")

    params = {
        "objective": "regression",
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "min_child_samples": config.min_child_samples,
        "seed": config.seed,
        "num_threads": config.num_threads,
        "deterministic": True,
        "verbosity": -1,
    }
    dataset = lgb.Dataset(train[list(config.feature_columns)], label=train[config.label_column])
    model = lgb.train(params, dataset, num_boost_round=config.n_estimators)

    test = test.copy()
    test["score"] = model.predict(test[list(config.feature_columns)])
    scores = test[["timestamp", "symbol", "score"]].reset_index(drop=True)
    candidates = _top_candidates(test, top_k)
    factor_report = analyze_factor_panel(
        test,
        factor_column="score",
        return_columns=[config.label_column],
        date_column="timestamp",
        asset_column="symbol",
    )
    metadata = _metadata(config, panel, model, oos_boundary, top_k)
    return BaselineResult(scores=scores, candidates=candidates, factor_report=factor_report, metadata=metadata)
