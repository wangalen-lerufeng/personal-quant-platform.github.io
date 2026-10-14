"""Qlib Alpha158 factor pipeline: real Qlib feature engine + models.

Uses Qlib's ``Alpha158`` handler (about 158 cross-sectional factors) by default,
or a custom ``DataHandlerLP`` with user-supplied expression factors. The data
must already be dumped to Qlib's ``.bin`` format via ``qlib_dump``.
"""

import pandas as pd

DEFAULT_LABEL = (["Ref($close, -1)/$close - 1"], ["LABEL0"])


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
    **model_kwargs,
) -> dict:
    """Train a Qlib model on Alpha158 (or custom) factors and return scores + candidates."""
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

    if feature_expressions:
        handler = _make_custom_handler(list(feature_expressions), list(symbols), start, end)
    else:
        handler = Alpha158(instruments=list(symbols), start_time=start, end_time=end, label=DEFAULT_LABEL)

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
