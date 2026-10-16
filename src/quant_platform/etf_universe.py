"""Deterministic ETF data-foundation primitives for the Phase 9 research pipeline.

Pure pandas transforms that build the cross-sectional panel the ML baseline
consumes: adjusted prices, a tradable pool, suspension/liquidity filters, and
explicit forward-return labels. No network I/O and no order/risk side effects.

The panel output matches `factor_analysis.analyze_factor_panel`'s contract
(date/asset/factor/forward-label columns). Forward returns are labels and use
future prices by design; feature construction must never consume them.
"""

import pandas as pd


def apply_adjustment_factor(daily: pd.DataFrame, adj_factors: pd.DataFrame) -> pd.DataFrame:
    """Merge raw daily prices with adjustment factors and add ``adj_close``.

    ``daily`` needs ``symbol``, ``trade_date``, ``close`` columns; ``adj_factors``
    needs ``symbol``, ``trade_date``, ``adj_factor``. Rows without a matching
    factor are dropped. ``adj_close = close * adj_factor`` (backward-adjusted).
    """
    for column in ("symbol", "trade_date", "close"):
        if column not in daily.columns:
            raise ValueError(f"daily is missing required column: {column}")
    for column in ("symbol", "trade_date", "adj_factor"):
        if column not in adj_factors.columns:
            raise ValueError(f"adj_factors is missing required column: {column}")
    merged = daily.merge(adj_factors, on=["symbol", "trade_date"], how="inner")
    merged["adj_close"] = merged["close"] * merged["adj_factor"]
    return merged


def tradable_pool(basics: list[dict], listed_status: str = "L") -> list[str]:
    """Return the sorted, de-duplicated ``ts_code`` list of listed ETFs."""
    codes = {
        str(row["ts_code"])
        for row in basics
        if row.get("list_status") == listed_status and row.get("ts_code")
    }
    return sorted(codes)


def drop_suspended(daily: pd.DataFrame, volume_column: str = "volume") -> pd.DataFrame:
    """Drop zero/missing-volume rows (suspended or no-trade days)."""
    if volume_column not in daily.columns:
        raise ValueError(f"daily is missing required volume column: {volume_column}")
    volume = pd.to_numeric(daily[volume_column], errors="coerce")
    return daily[volume.notna() & (volume > 0)].copy()


def liquid_symbols(
    daily: pd.DataFrame,
    window: int = 20,
    min_median_amount: float = 0.0,
    amount_column: str = "amount",
) -> list[str]:
    """Return symbols whose trailing median ``amount`` meets ``min_median_amount``."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if amount_column not in daily.columns:
        raise ValueError(f"daily is missing required amount column: {amount_column}")
    liquid = []
    for symbol, group in daily.groupby("symbol"):
        tail = pd.to_numeric(
            group.sort_values("trade_date")[amount_column].tail(window),
            errors="coerce",
        ).dropna()
        if tail.empty:
            continue
        if float(tail.median()) >= min_median_amount:
            liquid.append(str(symbol))
    return sorted(liquid)


def add_forward_returns(
    daily: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    price_column: str = "adj_close",
) -> pd.DataFrame:
    """Add ``forward_return_{h}d`` label columns per symbol (uses future prices)."""
    if price_column not in daily.columns:
        raise ValueError(f"daily is missing required price column: {price_column}")
    result = daily.copy()
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("horizons must be positive")
        shifted = result.groupby("symbol")[price_column].shift(-horizon)
        result[f"forward_return_{horizon}d"] = shifted / result[price_column] - 1
    return result


def build_etf_panel(
    daily: pd.DataFrame,
    adj_factors: pd.DataFrame,
    basics: list[dict] | None = None,
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    min_median_amount: float | None = None,
    amount_window: int = 20,
) -> pd.DataFrame:
    """Compose adjusted prices, filters, and labels into an analyzable panel.

    Output columns: ``timestamp, symbol, adj_close, volume, amount, forward_return_*``,
    sorted by ``(timestamp, symbol)`` and de-duplicated on ``(symbol, trade_date)``.
    """
    required = {"symbol", "trade_date", "close", "volume", "amount"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily is missing required columns: {', '.join(missing)}")

    panel = apply_adjustment_factor(daily, adj_factors)
    panel = drop_suspended(panel)
    if basics is not None:
        pool = set(tradable_pool(basics))
        panel = panel[panel["symbol"].isin(pool)]
    if min_median_amount is not None:
        liquid = set(
            liquid_symbols(panel, window=amount_window, min_median_amount=min_median_amount)
        )
        panel = panel[panel["symbol"].isin(liquid)]
    # Tushare returns rows newest-first; sort ascending so shift(-N) is the forward return.
    panel = panel.sort_values(["symbol", "trade_date"])
    panel = add_forward_returns(panel, horizons=horizons)
    panel = panel.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
    panel["timestamp"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    forward_columns = [f"forward_return_{horizon}d" for horizon in horizons]
    panel = panel[
        ["timestamp", "symbol", "adj_close", "volume", "amount", *forward_columns]
    ].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return panel
