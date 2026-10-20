"""Point-in-time research-universe helpers.

These helpers only use information available at each timestamp.  They are kept
outside the model code so universe construction can be tested independently.
"""

import pandas as pd


def build_dynamic_universe_index(
    ohlcv: pd.DataFrame,
    *,
    liquidity_window: int = 20,
    top_n: int = 100,
    min_history: int = 250,
    monthly: bool = True,
) -> pd.MultiIndex:
    """Return eligible ``(instrument, datetime)`` pairs without future data."""
    if liquidity_window < 1 or top_n < 1 or min_history < 1:
        raise ValueError("liquidity_window, top_n and min_history must be positive")
    required = {"symbol", "date", "amount"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV lacks required columns: {sorted(missing)}")

    frame = ohlcv.loc[:, ["symbol", "date", "amount"]].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.sort_values(["symbol", "date"])
    grouped = frame.groupby("symbol", sort=False)
    frame["history"] = grouped.cumcount() + 1
    frame["trailing_amount"] = grouped["amount"].transform(
        lambda values: values.rolling(liquidity_window, min_periods=liquidity_window).mean()
    )
    frame = frame[
        (frame["history"] >= min_history)
        & frame["trailing_amount"].notna()
        & (frame["trailing_amount"] > 0)
    ].copy()
    if frame.empty:
        return pd.MultiIndex.from_arrays([[], []], names=["instrument", "datetime"])

    frame["liquidity_rank"] = frame.groupby("date")["trailing_amount"].rank(
        method="first", ascending=False
    )
    frame = frame[frame["liquidity_rank"] <= top_n]
    if monthly and not frame.empty:
        calendar = frame[["date"]].drop_duplicates().copy()
        calendar["month"] = calendar["date"].dt.to_period("M")
        month_ends = set(calendar.groupby("month")["date"].max())
        frame = frame[frame["date"].isin(month_ends)]

    return pd.MultiIndex.from_frame(
        frame[["symbol", "date"]].rename(
            columns={"symbol": "instrument", "date": "datetime"}
        )
    )


def rank_scores_within_industry(scores: pd.DataFrame, industry_map: dict) -> pd.DataFrame:
    """Replace raw scores with point-in-time cross-sectional industry percentiles."""
    result = scores.copy()
    result["_industry"] = result["symbol"].map(industry_map).fillna("unknown")
    result["score"] = result.groupby(["timestamp", "_industry"])["score"].rank(
        method="average", pct=True
    )
    return result.drop(columns=["_industry"])
