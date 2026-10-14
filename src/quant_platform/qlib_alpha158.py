"""Qlib Alpha158 factor pipeline: real Qlib feature engine + models.

Uses Qlib's ``Alpha158`` handler (about 158 cross-sectional factors computed by
Qlib's expression engine) instead of the hand-written five features. The data
must already be dumped to Qlib's ``.bin`` format via ``qlib_dump``.
"""

import pandas as pd


def run_alpha158(
    symbols,
    start_date,
    end_date,
    *,
    qlib_dir,
    top_k: int = 10,
    model_type: str = "lgb",
    num_models: int = 5,
    **model_kwargs,
) -> dict:
    """Train a Qlib model on Alpha158 factors and return OOS scores + candidates."""
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.double_ensemble import DEnsembleModel
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    from .qlib_dump import init_qlib

    init_qlib(qlib_dir)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = int((end - start).days)
    train_end = start + pd.Timedelta(days=int(total_days * 0.7))
    valid_end = start + pd.Timedelta(days=int(total_days * 0.85))

    handler = Alpha158(instruments=list(symbols), start_time=start, end_time=end)
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (start, train_end),
            "valid": (train_end, valid_end),
            "test": (valid_end, end),
        },
    )

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
