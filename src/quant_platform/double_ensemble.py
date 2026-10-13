"""DoubleEnsemble-style noise-robust cross-sectional model for Phase 9.

A lightweight, in-house implementation of the two core ideas from the
DoubleEnsemble paper (Liang et al., 2021):

* **Sample reweighting (SR)** — after each base learner, samples with the
  highest training loss are down-weighted, reducing the influence of noisy
  labels on subsequent learners.
* **Feature selection (FS)** — after each base learner, the least-important
  features (by LightGBM gain) are dropped, shrinking the input for the next
  learner.

Final predictions average all base learners. Read-only research: no orders,
no risk-engine interaction, and features remain causally computed (no future
data). Uses the native LightGBM API, so scikit-learn is not required.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factor_analysis import analyze_factor_panel
from .ml_baseline import (
    DEFAULT_FEATURE_COLUMNS,
    BaselineResult,
    compute_cross_sectional_features,
)


def _import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError as error:
        raise ImportError(
            "lightgbm is required for DoubleEnsemble; install it with "
            "`python3 -m pip install lightgbm`"
        ) from error


@dataclass(frozen=True)
class DoubleEnsembleConfig:
    label_column: str = "forward_return_5d"
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    n_models: int = 5
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 100
    min_child_samples: int = 20
    seed: int = 42
    num_threads: int = 1
    sample_reweighting: bool = True
    noise_quantile: float = 0.9
    noisy_sample_weight: float = 0.1
    feature_selection: bool = True
    feature_drop_ratio: float = 0.2
    min_features: int = 2

    def __post_init__(self):
        if self.n_models < 1:
            raise ValueError("n_models must be at least 1")
        if not 0 < self.noise_quantile < 1:
            raise ValueError("noise_quantile must be in (0, 1)")
        if not 0 <= self.feature_drop_ratio < 1:
            raise ValueError("feature_drop_ratio must be in [0, 1)")
        if self.min_features < 1:
            raise ValueError("min_features must be at least 1")
        if self.noisy_sample_weight < 0:
            raise ValueError("noisy_sample_weight must be non-negative")


class DoubleEnsembleModel:
    """Iteratively trains base learners with sample reweighting and feature selection."""

    def __init__(self, config: DoubleEnsembleConfig):
        self.config = config
        self.models: list = []
        self.feature_masks: list[list[str]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DoubleEnsembleModel":
        lgb = _import_lightgbm()
        features = list(X.columns)
        sample_weight = np.ones(len(X), dtype=float)
        target = np.asarray(y, dtype=float)
        for round_index in range(self.config.n_models):
            params = {
                "objective": "regression",
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "min_child_samples": self.config.min_child_samples,
                "seed": self.config.seed + round_index,
                "num_threads": self.config.num_threads,
                "deterministic": True,
                "verbosity": -1,
            }
            dataset = lgb.Dataset(X[features], label=target, weight=sample_weight)
            booster = lgb.train(params, dataset, num_boost_round=self.config.n_estimators)
            self.models.append(booster)
            self.feature_masks.append(list(features))

            if self.config.sample_reweighting:
                prediction = booster.predict(X[features])
                loss = (prediction - target) ** 2
                threshold = float(np.quantile(loss, self.config.noise_quantile))
                sample_weight = np.where(
                    loss >= threshold,
                    sample_weight * self.config.noisy_sample_weight,
                    sample_weight,
                )

            if self.config.feature_selection and len(features) > self.config.min_features:
                importances = booster.feature_importance("gain")
                n_drop = max(1, int(len(features) * self.config.feature_drop_ratio))
                if len(features) - n_drop >= self.config.min_features:
                    drop_indices = set(np.argsort(importances)[:n_drop].tolist())
                    features = [f for i, f in enumerate(features) if i not in drop_indices]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = [
            model.predict(X[features]) for model, features in zip(self.models, self.feature_masks)
        ]
        return np.mean(np.column_stack(predictions), axis=1)


def _top_candidates(frame: pd.DataFrame, top_k: int) -> dict:
    candidates = {}
    for timestamp, section in frame.groupby("timestamp"):
        top = section.sort_values("score", ascending=False)["symbol"].head(top_k)
        candidates[pd.Timestamp(timestamp).isoformat()] = [str(symbol) for symbol in top]
    return candidates


def _metadata(config, panel, model, oos_boundary, top_k) -> dict:
    key = panel[["timestamp", "symbol", "adj_close", "volume", "amount"]].sort_values(
        ["timestamp", "symbol"]
    )
    data_fingerprint = hashlib.sha256(key.to_csv(index=False).encode("utf-8")).hexdigest()
    model_string = "\n".join(booster.model_to_string() for booster in model.models)
    model_signature = hashlib.sha256(model_string.encode("utf-8")).hexdigest()
    return {
        "model": "double_ensemble",
        "label_column": config.label_column,
        "n_models": config.n_models,
        "hyperparameters": {
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "n_estimators": config.n_estimators,
            "min_child_samples": config.min_child_samples,
        },
        "seed": config.seed,
        "num_threads": config.num_threads,
        "sample_reweighting": config.sample_reweighting,
        "noise_quantile": config.noise_quantile,
        "noisy_sample_weight": config.noisy_sample_weight,
        "feature_selection": config.feature_selection,
        "feature_drop_ratio": config.feature_drop_ratio,
        "min_features": config.min_features,
        "top_k": top_k,
        "train_until": oos_boundary.isoformat(),
        "data_fingerprint": data_fingerprint,
        "model_signature": model_signature,
        "feature_selection_history": [list(features) for features in model.feature_masks],
        "final_feature_set": list(model.feature_masks[-1]),
        "research_only": True,
        "execution_enabled": False,
    }


def run_double_ensemble(
    panel: pd.DataFrame,
    *,
    config: DoubleEnsembleConfig | None = None,
    out_of_sample_start=None,
    top_k: int = 10,
) -> BaselineResult:
    """Train a DoubleEnsemble model and evaluate it out-of-sample."""
    config = config or DoubleEnsembleConfig()
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

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

    model = DoubleEnsembleModel(config)
    model.fit(train[list(config.feature_columns)], train[config.label_column])

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
    return BaselineResult(
        scores=scores, candidates=candidates, factor_report=factor_report, metadata=metadata,
    )
