"""Causal, reproducible technical feature calculations."""

import pandas as pd


def add_basic_features(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")
    required = {"close", "high", "low", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing feature input columns: {sorted(missing)}")
    result = frame.copy()
    result = result.sort_values("timestamp" if "timestamp" in result else result.index).reset_index(drop=True)
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    result["return_1d"] = close.pct_change()
    result[f"momentum_{window}d"] = close.pct_change(window)
    result[f"volatility_{window}d"] = result["return_1d"].rolling(window).std()
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    result[f"atr_{window}d"] = true_range.rolling(window).mean()
    result[f"ma_distance_{window}d"] = close / close.rolling(window).mean() - 1
    result[f"volume_ratio_{window}d"] = result["volume"].astype(float) / result["volume"].rolling(window).mean()
    return result
