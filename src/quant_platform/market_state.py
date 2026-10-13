"""Explainable market-regime classification for research only."""

import pandas as pd


def classify_market_state(frame: pd.DataFrame, window: int = 20) -> dict:
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if len(frame) < window:
        return {"state": "insufficient_data", "evidence": {"rows": len(frame), "window": window}}
    close = pd.to_numeric(frame["close"], errors="coerce")
    returns = close.pct_change().dropna()
    volatility = float(returns.tail(window).std())
    trend = float(close.iloc[-1] / close.iloc[-window] - 1)
    threshold = float(returns.tail(window).std() * 2)
    if volatility > 0 and abs(trend) < threshold:
        state = "high_volatility"
    elif abs(trend) >= threshold:
        state = "trend"
    else:
        state = "range"
    return {"state": state, "evidence": {"window": window, "trend_return": trend, "volatility": volatility, "trend_threshold": threshold}}
