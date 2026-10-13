"""Small, explicit factor analytics primitives for research reports.

Panel analytics intentionally consume *precomputed* forward-return labels.  The
module never shifts prices itself, which keeps label construction auditable and
prevents an accidental same-bar/future-data shortcut in this research layer.
"""

import math

import pandas as pd


def winsorize_zscore(values: pd.Series, limits: tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    if len(limits) != 2 or min(limits) < 0 or sum(limits) >= 1:
        raise ValueError("limits must be non-negative and sum to less than 1")
    clean = pd.to_numeric(values, errors="coerce")
    lower, upper = clean.quantile(limits[0]), clean.quantile(1 - limits[1])
    clipped = clean.clip(lower, upper)
    std = clipped.std(ddof=0)
    if pd.isna(std) or std == 0:
        return clipped * 0.0
    return (clipped - clipped.mean()) / std


def analyze_factor(frame: pd.DataFrame, factor_column: str, return_column: str, groups: int = 5) -> dict:
    if groups < 2:
        raise ValueError("groups must be at least 2")
    if factor_column not in frame or return_column not in frame:
        raise ValueError("factor and forward-return columns are required")
    sample = frame[[factor_column, return_column]].dropna().copy()
    if len(sample) < groups:
        raise ValueError("not enough observations for factor grouping")
    sample["group"] = pd.qcut(sample[factor_column], q=groups, labels=False, duplicates="drop")
    group_returns = sample.groupby("group", observed=True)[return_column].mean().to_dict()
    rank_factor = sample[factor_column].rank(method="average")
    rank_return = sample[return_column].rank(method="average")
    return {
        "observations": int(len(sample)),
        "ic": float(sample[factor_column].corr(sample[return_column], method="pearson")),
        "rank_ic": float(rank_factor.corr(rank_return, method="pearson")),
        "group_returns": {str(int(group)): float(value) for group, value in group_returns.items()},
    }


def _finite(value: float) -> float | None:
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def _cross_section_metrics(
    sample: pd.DataFrame,
    date_column: str,
    factor_column: str,
    return_column: str,
) -> tuple[list[dict], dict]:
    series: list[dict] = []
    for timestamp, section in sample.groupby(date_column, sort=True):
        valid = section[[factor_column, return_column]].dropna()
        if len(valid) < 2 or valid[factor_column].nunique() < 2 or valid[return_column].nunique() < 2:
            continue
        ic = valid[factor_column].corr(valid[return_column], method="pearson")
        rank_ic = valid[factor_column].rank().corr(valid[return_column].rank(), method="pearson")
        series.append({
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "observations": int(len(valid)),
            "ic": _finite(ic),
            "rank_ic": _finite(rank_ic),
        })
    ic_values = pd.Series([row["ic"] for row in series if row["ic"] is not None], dtype=float)
    rank_values = pd.Series([row["rank_ic"] for row in series if row["rank_ic"] is not None], dtype=float)
    ic_mean = ic_values.mean() if not ic_values.empty else float("nan")
    ic_std = ic_values.std(ddof=1) if len(ic_values) > 1 else float("nan")
    return series, {
        "date_count": int(len(series)),
        "ic_mean": _finite(ic_mean),
        "ic_std": _finite(ic_std),
        "icir": _finite(ic_mean / ic_std) if pd.notna(ic_std) and ic_std != 0 else None,
        "rank_ic_mean": _finite(rank_values.mean()) if not rank_values.empty else None,
        "rank_ic_std": _finite(rank_values.std(ddof=1)) if len(rank_values) > 1 else None,
    }


def _factor_autocorrelation(
    sample: pd.DataFrame,
    date_column: str,
    asset_column: str,
    factor_column: str,
) -> float | None:
    dates = list(sample[date_column].drop_duplicates().sort_values())
    correlations: list[float] = []
    for previous_date, current_date in zip(dates, dates[1:]):
        previous = sample.loc[sample[date_column] == previous_date, [asset_column, factor_column]]
        current = sample.loc[sample[date_column] == current_date, [asset_column, factor_column]]
        paired = previous.merge(current, on=asset_column, suffixes=("_previous", "_current")).dropna()
        if len(paired) < 2:
            continue
        correlation = paired[f"{factor_column}_previous"].rank().corr(
            paired[f"{factor_column}_current"].rank()
        )
        if pd.notna(correlation):
            correlations.append(float(correlation))
    return _finite(pd.Series(correlations).mean()) if correlations else None


def _top_group_statistics(
    sample: pd.DataFrame,
    date_column: str,
    asset_column: str,
    factor_column: str,
    groups: int,
) -> tuple[float | None, pd.Series]:
    memberships: list[set[str]] = []
    top_rows: list[pd.DataFrame] = []
    for _timestamp, section in sample.groupby(date_column, sort=True):
        valid = section.dropna(subset=[factor_column]).copy()
        if len(valid) < groups:
            continue
        valid["_factor_group"] = pd.qcut(
            valid[factor_column], q=groups, labels=False, duplicates="drop"
        )
        if valid["_factor_group"].isna().all():
            continue
        top = valid[valid["_factor_group"] == valid["_factor_group"].max()]
        memberships.append(set(top[asset_column].astype(str)))
        top_rows.append(top)
    turnovers = []
    for previous, current in zip(memberships, memberships[1:]):
        if previous:
            turnovers.append(1 - len(previous & current) / len(previous))
    combined = pd.concat(top_rows, ignore_index=True) if top_rows else sample.iloc[0:0].copy()
    return (_finite(pd.Series(turnovers).mean()) if turnovers else None), combined


def _sample_period_summary(
    sample: pd.DataFrame,
    date_column: str,
    factor_column: str,
    return_columns: list[str],
) -> dict:
    horizons = {}
    for return_column in return_columns:
        _series, summary = _cross_section_metrics(sample, date_column, factor_column, return_column)
        horizons[return_column] = summary
    return {
        "observations": int(len(sample)),
        "date_count": int(sample[date_column].nunique()),
        "horizons": horizons,
    }


def analyze_factor_panel(
    frame: pd.DataFrame,
    factor_column: str,
    return_columns: list[str] | tuple[str, ...],
    groups: int = 5,
    date_column: str = "timestamp",
    asset_column: str = "symbol",
    out_of_sample_start: str | pd.Timestamp | None = None,
    dollar_volume_column: str | None = None,
    participation_rate: float = 0.01,
) -> dict:
    """Analyze date-by-asset factor observations against explicit future labels."""
    if groups < 2:
        raise ValueError("groups must be at least 2")
    if not return_columns:
        raise ValueError("at least one forward-return column is required")
    return_columns = [str(column) for column in return_columns]
    if any("forward" not in column.lower() for column in return_columns):
        raise ValueError("return columns must be explicit forward-return labels")
    required = {date_column, asset_column, factor_column, *return_columns}
    if dollar_volume_column:
        required.add(dollar_volume_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")

    sample = frame[list(required)].copy()
    sample[date_column] = pd.to_datetime(sample[date_column], errors="coerce", utc=True).dt.tz_convert(None)
    sample[factor_column] = pd.to_numeric(sample[factor_column], errors="coerce")
    for column in return_columns:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    if dollar_volume_column:
        sample[dollar_volume_column] = pd.to_numeric(sample[dollar_volume_column], errors="coerce")
    sample = sample.dropna(subset=[date_column, asset_column, factor_column]).sort_values(
        [date_column, asset_column]
    )
    if sample.duplicated([date_column, asset_column]).any():
        raise ValueError("duplicate asset observations exist for the same timestamp")
    if sample[date_column].nunique() < 2:
        raise ValueError("factor panel requires at least two timestamps")

    ic_series: dict[str, list[dict]] = {}
    horizons: dict[str, dict] = {}
    group_returns: dict[str, dict[str, float]] = {}
    for return_column in return_columns:
        series, summary = _cross_section_metrics(sample, date_column, factor_column, return_column)
        ic_series[return_column] = series
        horizons[return_column] = summary
        valid = sample[[factor_column, return_column]].dropna().copy()
        valid["group"] = pd.qcut(valid[factor_column], q=groups, labels=False, duplicates="drop")
        means = valid.groupby("group", observed=True)[return_column].mean()
        group_returns[return_column] = {
            str(int(group)): float(value) for group, value in means.items()
        }

    turnover, top_group = _top_group_statistics(
        sample, date_column, asset_column, factor_column, groups
    )
    capacity = None
    if dollar_volume_column:
        daily_top_liquidity = top_group.groupby(date_column)[dollar_volume_column].sum(min_count=1)
        estimated = daily_top_liquidity.median() * participation_rate
        capacity = {
            "method": "median_top_group_dollar_volume_x_participation_rate",
            "participation_rate": participation_rate,
            "estimated_notional": _finite(estimated),
            "currency": "source_data_currency",
        }

    split = None
    if out_of_sample_start is not None:
        boundary = pd.Timestamp(out_of_sample_start)
        in_sample = sample[sample[date_column] < boundary]
        out_sample = sample[sample[date_column] >= boundary]
        if in_sample.empty or out_sample.empty:
            raise ValueError("out_of_sample_start must leave non-empty in-sample and out-of-sample periods")
        split = {
            "out_of_sample_start": boundary.isoformat(),
            "in_sample": _sample_period_summary(in_sample, date_column, factor_column, return_columns),
            "out_of_sample": _sample_period_summary(out_sample, date_column, factor_column, return_columns),
        }

    return {
        "observations": int(len(sample)),
        "date_count": int(sample[date_column].nunique()),
        "horizons": horizons,
        "ic_series": ic_series,
        "decay": {column: horizons[column]["rank_ic_mean"] for column in return_columns},
        "group_returns": group_returns,
        "factor_autocorrelation": _factor_autocorrelation(
            sample, date_column, asset_column, factor_column
        ),
        "top_group_turnover": turnover,
        "capacity": capacity,
        "sample_split": split,
        "label_columns": return_columns,
        "alignment": "factor_at_t_vs_forward_return_after_t",
        "future_data_used_for_feature": False,
    }
