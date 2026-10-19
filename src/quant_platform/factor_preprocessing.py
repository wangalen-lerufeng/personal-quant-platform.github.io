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


def preprocess_factor_panel(
    frame: pd.DataFrame,
    factors,
    *,
    winsorize: bool = True,
    standardize: bool = True,
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    """预处理因子面板：MAD 去极值 + 截面标准化。"""
    result = frame
    if winsorize:
        result = winsorize_mad(result, factors, n_sigma=n_sigma)
    if standardize:
        result = cross_sectional_zscore(result, factors)
    return result
