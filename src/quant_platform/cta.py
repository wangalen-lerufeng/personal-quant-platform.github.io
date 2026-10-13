"""Causal single-asset CTA signal primitives."""

import pandas as pd


def donchian_signal(frame: pd.DataFrame, entry_window: int = 55, exit_window: int = 20) -> pd.DataFrame:
    if entry_window <= 0 or exit_window <= 0:
        raise ValueError("windows must be positive")
    required = {"high", "low", "close"}
    if not required.issubset(frame.columns):
        raise ValueError("high, low and close columns are required")
    result = frame.copy()
    result["entry_high"] = result["high"].rolling(entry_window).max().shift(1)
    result["entry_low"] = result["low"].rolling(entry_window).min().shift(1)
    result["exit_high"] = result["high"].rolling(exit_window).max().shift(1)
    result["exit_low"] = result["low"].rolling(exit_window).min().shift(1)
    result["signal"] = 0
    result.loc[result["close"] > result["entry_high"], "signal"] = 1
    result.loc[result["close"] < result["entry_low"], "signal"] = -1
    return result
