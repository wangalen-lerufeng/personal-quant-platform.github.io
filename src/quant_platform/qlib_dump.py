"""Dump ETF OHLCV data into Qlib's ``.bin`` format for the Alpha158 pipeline.

Qlib's feature engine reads raw fields (open/high/low/close/vwap/volume/amount)
from ``<instrument>/<field>.day.bin`` files. Each file is a little-endian
float32 array: the first element is the instrument's start index in the global
trading calendar, followed by the field values aligned to that calendar.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .providers import TushareDatasetProvider

FIELDS = ["open", "high", "low", "close", "vwap", "volume", "amount"]


def init_qlib(provider_uri) -> None:
    """Initialize Qlib, ending any previously-active experiment first."""
    import qlib
    from qlib.workflow import R

    try:
        R.end_exp()
    except Exception:
        pass
    qlib.init(provider_uri=provider_uri, region="cn")


def _fetch_symbol_ohlcv(symbol, provider, daily_dataset, start_date, end_date):
    # Tushare A-share uses amount in 千元 and vol in 手 (100 shares/lots), so the
    # per-share VWAP is amount/vol * 10. HK daily is already in per-share units.
    vwap_factor = 10.0 if daily_dataset in ("fund_daily", "daily") else 1.0
    rows = []
    for record in provider.fetch_records(daily_dataset, symbol, start_date, end_date):
        close = record.get("close")
        if close is None:
            continue
        close = float(close)
        volume = float(record.get("vol") or 0.0)
        amount = float(record.get("amount") or 0.0)
        rows.append({
            "symbol": symbol,
            "date": str(record.get("trade_date", "")).replace("-", "")[:8],
            "open": float(record.get("open") or close),
            "high": float(record.get("high") or close),
            "low": float(record.get("low") or close),
            "close": close,
            "vwap": amount / volume * vwap_factor if volume > 0 else close,
            "volume": volume,
            "amount": amount,
        })
    return rows


def build_ohlcv_panel(symbols, start_date, end_date, token_env="TUSHARE_TOKEN", asset_class="etf", max_workers=1) -> pd.DataFrame:
    """Fetch OHLCV + amount from Tushare (``fund_daily`` or ``daily``) and derive VWAP."""
    symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not symbols:
        raise ValueError("symbols must be non-empty")
    daily_dataset = "fund_daily" if asset_class == "etf" else ("hk_daily" if asset_class == "hk" else "daily")
    provider = TushareDatasetProvider(token_env)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda symbol: _fetch_symbol_ohlcv(symbol, provider, daily_dataset, start_date, end_date),
            symbols,
        ))
    rows = [row for sublist in results for row in sublist]
    if not rows:
        raise ValueError(f"no Tushare {daily_dataset} data returned")
    return pd.DataFrame(rows)


def dump_etf_to_qlib(frame: pd.DataFrame, qlib_dir) -> str:
    """Write an OHLCV frame to Qlib's ``.bin`` format and return the dir path."""
    qlib_dir = Path(qlib_dir).expanduser()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    calendar = sorted(frame["date"].unique())

    calendars_dir = qlib_dir / "calendars"
    instruments_dir = qlib_dir / "instruments"
    features_dir = qlib_dir / "features"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)

    with (calendars_dir / "day.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(date.strftime("%Y-%m-%d") for date in calendar) + "\n")

    instrument_lines = []
    for symbol in sorted(frame["symbol"].unique()):
        sub = frame[frame["symbol"] == symbol]
        instrument_lines.append(
            f"{symbol}\t{sub['date'].min().strftime('%Y-%m-%d')}\t{sub['date'].max().strftime('%Y-%m-%d')}"
        )
    with (instruments_dir / "all.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(instrument_lines) + "\n")

    for symbol in sorted(frame["symbol"].unique()):
        sub = frame[frame["symbol"] == symbol].set_index("date")
        first_date = sub.index.min()
        last_date = sub.index.max()
        cal_sub = [date for date in calendar if first_date <= date <= last_date]
        aligned = sub.reindex(cal_sub)
        date_index = calendar.index(first_date)

        field_dir = features_dir / symbol.lower()
        field_dir.mkdir(parents=True, exist_ok=True)
        for field in FIELDS:
            values = aligned[field].to_numpy(dtype="<f")
            array = np.hstack([np.array([date_index], dtype="<f"), values]).astype("<f")
            array.tofile(str(field_dir / f"{field}.day.bin"))

    return str(qlib_dir)
