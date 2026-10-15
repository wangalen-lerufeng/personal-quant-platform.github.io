"""Qlib Alpha158 factor pipeline: real Qlib feature engine + models.

Uses Qlib's ``Alpha158`` handler (about 158 cross-sectional factors) by default,
or a custom ``DataHandlerLP`` with user-supplied expression factors. The data
must already be dumped to Qlib's ``.bin`` format via ``qlib_dump``.
"""

import pandas as pd

DEFAULT_LABEL = (["Ref($close, -1)/$close - 1"], ["LABEL0"])
FORWARD_RETURN_EXPR = "Ref($close, -1)/$close - 1"


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
