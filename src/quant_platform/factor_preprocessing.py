"""因子预处理：缺失值检查、截面去极值、截面标准化。

在 Qlib 计算原始因子之后、LightGBM/DoubleEnsemble 训练之前执行。所有操作都按
交易日横截面（groupby datetime）进行，不跨时间混合统计量。
"""

import numpy as np
import pandas as pd


def missing_report(frame: pd.DataFrame, factors) -> dict:
    """返回每个因子的缺失率，以及每个日期的平均有效股票数。"""
    sub = frame[factors]
    factor_missing = {f: float(sub[f].isna().mean()) for f in factors}
    per_date = sub.notna().sum(axis=1).groupby(level="datetime").mean()
    return {
        "factor_missing_ratio": factor_missing,
        "mean_valid_per_date": float(per_date.mean()) if len(per_date) else None,
    }


def winsorize_mad(frame: pd.DataFrame, factors, n_sigma: float = 3.0) -> pd.DataFrame:
    """按交易日横截面对每个因子做 MAD 去极值。

    上/下界 = 中位数 ± n_sigma × 1.4826 × MAD（MAD 为绝对中位差）。
    """
    result = frame.copy()
    for f in factors:
        def _winsorize(s: pd.Series) -> pd.Series:
            median = s.median()
            mad = (s - median).abs().median()
            if not mad or pd.isna(mad):
                return s
            bound = n_sigma * 1.4826 * mad
            return s.clip(lower=median - bound, upper=median + bound)

        result[f] = result[f].groupby(level="datetime").transform(_winsorize)
    return result


def cross_sectional_zscore(frame: pd.DataFrame, factors) -> pd.DataFrame:
    """按交易日横截面对每个因子做 Z-Score 标准化。"""
    result = frame.copy()
    for f in factors:
        result[f] = result[f].groupby(level="datetime").transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12)
        )
    return result


def neutralize(
    frame: pd.DataFrame,
    factors,
    industry_map: dict,
    log_mv: pd.Series,
) -> pd.DataFrame:
    """行业 + 市值中性化：每日横截面回归取残差。

    回归目标：``因子值 = 截距 + 行业哑变量 + log(总市值) + 残差``，用残差替代因子，
    避免把行业暴露或小市值暴露误认为 Alpha。

    ``log_mv`` 需要以 ``(instrument, datetime)`` 为索引。
    """
    result = frame.copy()
    instruments = result.index.get_level_values("instrument")
    result["_industry"] = [industry_map.get(i, "unknown") for i in instruments]
    result["_log_mv"] = log_mv.reindex(result.index).astype(float)

    for f in factors:
        resid = pd.Series(np.nan, index=result.index, dtype=float)
        for date, group in result.groupby(level="datetime"):
            y = group[f].to_numpy(dtype=float)
            dummies = pd.get_dummies(group["_industry"], drop_first=True).to_numpy(dtype=float)
            X = np.column_stack([
                np.ones(len(group)),
                dummies,
                group["_log_mv"].to_numpy(dtype=float),
            ])
            mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            if mask.sum() > X.shape[1]:
                beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
                r = y - X @ beta
                r[~mask] = np.nan
                resid.loc[group.index] = r
        result[f] = resid

    return result.drop(columns=["_industry", "_log_mv"])


def preprocess_factor_panel(
    frame: pd.DataFrame,
    factors,
    *,
    winsorize: bool = True,
    standardize: bool = True,
    n_sigma: float = 3.0,
    industry_map: dict | None = None,
    log_mv: pd.Series | None = None,
) -> pd.DataFrame:
    """预处理因子面板：MAD 去极值 → 行业市值中性化 → 截面标准化。"""
    result = frame
    if winsorize:
        result = winsorize_mad(result, factors, n_sigma=n_sigma)
    if industry_map is not None and log_mv is not None:
        result = neutralize(result, factors, industry_map, log_mv)
    if standardize:
        result = cross_sectional_zscore(result, factors)
    return result
