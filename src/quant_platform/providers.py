"""Optional online market-data providers."""
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from http.client import IncompleteRead
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests
from urllib.parse import urlsplit
from .domain import AssetClass, MarketEvent
from .data_quality import validate_events


_TUSHARE_REQUESTS_PER_MINUTE = 240  # 留出余量，避免触发 Tushare 300次/分钟上限


class _RateLimiter:
    """进程级限流：让 Tushare 请求均匀铺开，避免批量取数时触发频率上限。"""

    def __init__(self, requests_per_minute: int):
        self._lock = threading.Lock()
        self._interval = 60.0 / requests_per_minute
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
            self._next = time.monotonic() + self._interval


_tushare_rate_limiter = _RateLimiter(_TUSHARE_REQUESTS_PER_MINUTE)


_TUSHARE_CACHE_DIR = Path("data/tushare_cache")


def _cache_key_full(dataset: str, symbol: str) -> str:
    return hashlib.md5(f"{dataset}|{symbol}".encode("utf-8")).hexdigest()


def _cache_load_full(dataset: str, symbol: str):
    path = _TUSHARE_CACHE_DIR / f"{_cache_key_full(dataset, symbol)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_store_full(dataset: str, symbol: str, records) -> None:
    path = _TUSHARE_CACHE_DIR / f"{_cache_key_full(dataset, symbol)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _record_date(record) -> str:
    return str(record.get("trade_date", "")).replace("-", "")[:8]


def _advance_days(yyyymmdd: str, days: int) -> str:
    return (datetime.strptime(yyyymmdd, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


def _filter_records(records, start_yyyymmdd: str, end_yyyymmdd: str):
    result = []
    for record in records:
        day = _record_date(record)
        if not day or start_yyyymmdd <= day <= end_yyyymmdd:
            result.append(record)
    return result


def _merge_records(base, new):
    """按整条记录去重合并，并按键（trade_date）排序。"""
    merged = {}
    for record in list(base) + list(new):
        merged[json.dumps(record, sort_keys=True, ensure_ascii=False)] = record
    records = list(merged.values())
    records.sort(key=_record_date)
    return records


def _bypass_proxy_for_tushare() -> None:
    """Exclude Tushare (a mainland API) from any local HTTP proxy.

    Tushare uses ``requests`` which honours ``HTTP(S)_PROXY``; routing it through a
    local Clash proxy can time out. Add its hosts to ``NO_PROXY`` so it connects
    directly.
    """
    entries = [d.strip() for d in os.environ.get("NO_PROXY", "").split(",") if d.strip()]
    for domain in (".tushare.pro", ".waditu.com"):
        if domain not in entries:
            entries.append(domain)
    value = ",".join(entries)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _create_tushare_client(token_env: str):
    token = (os.getenv(token_env) or "").strip()
    if not token:
        raise RuntimeError(f"Tushare token is not configured in {token_env}")
    _bypass_proxy_for_tushare()
    try:
        import tushare as ts
    except ImportError as error:
        raise RuntimeError("Tushare is not installed; run python3 -m pip install tushare") from error
    client = ts.pro_api(token)
    base_url = (os.getenv("TUSHARE_BASE_URL") or "").strip()
    if base_url:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("TUSHARE_BASE_URL must be an http or https URL")
        if parsed.query or parsed.fragment:
            raise RuntimeError("TUSHARE_BASE_URL must not contain query parameters or fragments")
        client._DataApi__http_url = base_url.rstrip("/") + "/"
    return client


class AkshareForexProvider:
    def fetch(self, symbol: str, start_date: str, end_date: str) -> list[MarketEvent]:
        try:
            return self._fetch_eastmoney_direct(symbol, start_date, end_date)
        except requests.Timeout as error:
            raise RuntimeError(
                "Eastmoney market-data request timed out after 20 seconds; "
                "verify Clash HTTP proxy 7899 and the Eastmoney API endpoint"
            ) from error
        except requests.RequestException as error:
            raise RuntimeError(
                f"Eastmoney direct market-data request failed: {error}; "
                "Eastmoney is unavailable for this EURUSD endpoint. "
                "Use MT5 broker history or import a CSV data file instead."
            ) from error
        except (KeyError, TypeError, ValueError) as direct_error:
            # Only malformed/unsupported direct responses fall back to AkShare.
            self._direct_error = direct_error
        try:
            import akshare as ak
        except ImportError as error:
            raise RuntimeError("AkShare is not installed; run python3 -m pip install akshare") from error
        # AkShare 1.18.x accepts only symbol; apply the requested date range locally.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            for attempt in range(3):
                try:
                    future = executor.submit(ak.forex_hist_em, symbol=symbol)
                    frame = future.result(timeout=30)
                    break
                except IncompleteRead as error:
                    if attempt == 2:
                        raise RuntimeError("AkShare response was truncated after 3 attempts; check Clash HTTP proxy 7899") from error
                    time.sleep(1)
        except FuturesTimeoutError as error:
            future.cancel()
            raise RuntimeError("AkShare market-data request timed out after 30 seconds") from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if frame is None or frame.empty:
            return []
        columns = {str(column).lower(): column for column in frame.columns}
        date_column = next((columns[name] for name in ("date", "日期", "datetime") if name in columns), None)
        close_column = next((columns[name] for name in ("close", "收盘", "收盘价", "最新价") if name in columns), None)
        if date_column is None or close_column is None:
            raise ValueError("AkShare response lacks date/close columns")
        start = start_date[:10]
        end = end_date[:10]
        frame = frame[(frame[date_column].astype(str).str[:10] >= start) & (frame[date_column].astype(str).str[:10] <= end)]
        events = [MarketEvent(symbol, AssetClass.FX, float(row[close_column]), datetime.fromisoformat(str(row[date_column])[:10] + "T00:00:00+00:00")) for _, row in frame.iterrows()]
        validate_events(events)
        return events

    @staticmethod
    def _fetch_eastmoney_direct(symbol: str, start_date: str, end_date: str) -> list[MarketEvent]:
        market_code = {"EURUSD": "119", "USDCNH": "119"}.get(symbol)
        if market_code is None:
            raise ValueError(f"unsupported Eastmoney FX symbol: {symbol}")
        # Eastmoney is a mainland source; bypass HTTP(S)_PROXY explicitly.
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={"secid": f"{market_code}.{symbol}", "klt": "101", "fqt": "1", "lmt": "50000", "end": "20500000", "iscca": "1", "fields1": "f1,f2,f3,f4,f5,f6,f7,f8", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64", "ut": "f057cbcbce2a86e2866ab8877db1d059", "forcect": 1},
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json()["data"]["klines"]
        events = []
        for row in rows:
            fields = row.split(",")
            timestamp = fields[0][:10]
            if start_date[:10] <= timestamp <= end_date[:10]:
                events.append(MarketEvent(symbol, AssetClass.FX, float(fields[2]), datetime.fromisoformat(timestamp + "T00:00:00+00:00")))
        validate_events(events)
        return events


class TushareProvider:
    """Read-only Tushare Pro provider for mainland ETF daily research data."""

    def __init__(self, token_env: str = "TUSHARE_TOKEN"):
        self.token_env = token_env

    def fetch(self, symbol: str, start_date: str, end_date: str) -> list[MarketEvent]:
        pro = _create_tushare_client(self.token_env)
        frame = pro.fund_daily(
            ts_code=symbol,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if frame is None or frame.empty:
            return []
        events = [
            MarketEvent(
                symbol,
                AssetClass.ETF,
                float(row["close"]),
                datetime.fromisoformat(str(row["trade_date"])[:10] + "T00:00:00+00:00"),
            )
            for _, row in frame.iterrows()
            if row.get("close") is not None
        ]
        events.sort(key=lambda event: event.timestamp)
        validate_events(events)
        return events


class TushareStockDailyProvider:
    """Read-only provider for the basic-access A-share ``daily`` endpoint."""

    def __init__(self, token_env: str = "TUSHARE_TOKEN"):
        self.token_env = token_env

    def fetch(self, symbol: str, start_date: str, end_date: str) -> list[MarketEvent]:
        frame = _create_tushare_client(self.token_env).daily(
            ts_code=symbol,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if frame is None or frame.empty:
            return []
        events = [
            MarketEvent(
                symbol,
                AssetClass.EQUITY,
                float(row["close"]),
                datetime.fromisoformat(str(row["trade_date"])[:10] + "T00:00:00+00:00"),
            )
            for _, row in frame.iterrows()
            if row.get("close") is not None
        ]
        events.sort(key=lambda event: event.timestamp)
        validate_events(events)
        return events


class TushareDatasetProvider:
    """Strict read-only dispatcher for non-price Tushare research datasets."""

    ALLOWED_DATASETS = {
        "stock_basic",
        "daily_basic",
        "adj_factor",
        "income",
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "forecast",
        "express",
        "moneyflow",
        "anns_d",
        "fund_basic",
        "fund_adj",
        "fund_share",
        "fund_daily",
        "daily",
        "index_weight",
        "index_daily",
        "hk_daily",
    }

    def __init__(self, token_env: str = "TUSHARE_TOKEN"):
        self.token_env = token_env

    def fetch_records(
        self, dataset: str, symbol: str, start_date: str, end_date: str
    ) -> list[dict]:
        """Fetch records, reusing a full-history local cache and only pulling the tail."""
        dataset = str(dataset).lower()
        if dataset not in self.ALLOWED_DATASETS:
            raise ValueError(f"unsupported Tushare dataset: {dataset}")
        start_yyyymmdd = start_date.replace("-", "")[:8]
        end_yyyymmdd = end_date.replace("-", "")[:8]

        cached = _cache_load_full(dataset, symbol)
        base = cached if cached is not None else []
        fetch_start = start_yyyymmdd

        if cached:
            dates = [day for day in (_record_date(record) for record in cached) if day]
            if dates:
                cmin, cmax = min(dates), max(dates)
                if cmin <= start_yyyymmdd and cmax >= end_yyyymmdd:
                    # 缓存已覆盖整个区间，直接切片返回
                    return _filter_records(cached, start_yyyymmdd, end_yyyymmdd)
                if cmin <= start_yyyymmdd:
                    # 只缺尾部，从缓存最新日期的次日开始增量拉取
                    fetch_start = _advance_days(cmax, 1)

        new_records = self._fetch_tushare(dataset, symbol, fetch_start, end_yyyymmdd)
        merged = _merge_records(base, new_records)
        if merged:
            _cache_store_full(dataset, symbol, merged)
        return _filter_records(merged, start_yyyymmdd, end_yyyymmdd)

    def _fetch_tushare(
        self, dataset: str, symbol: str, start_yyyymmdd: str, end_yyyymmdd: str
    ) -> list[dict]:
        """Raw rate-limited Tushare request for the given date range."""
        client = _create_tushare_client(self.token_env)
        method = getattr(client, dataset)
        dates = {"start_date": start_yyyymmdd, "end_date": end_yyyymmdd}
        if dataset == "stock_basic":
            arguments = {"ts_code": symbol}
        elif dataset == "fund_basic":
            arguments = {"ts_code": symbol, "market": "E"}
        elif dataset == "index_weight":
            arguments = {"index_code": symbol, **dates}
        else:
            arguments = {"ts_code": symbol, **dates}
        for attempt in range(3):
            try:
                _tushare_rate_limiter.acquire()
                frame = method(**arguments)
                break
            except Exception as error:
                message = str(error)
                if "频率超限" in message or "超限" in message:
                    if attempt == 2:
                        raise RuntimeError(
                            f"Tushare {dataset} 请求频率超限（300次/分钟），请缩小股票池或稍后重试: {message}"
                        ) from error
                    time.sleep(20)
                else:
                    raise RuntimeError(f"Tushare {dataset} 请求失败: {message}") from error
        if frame is None or frame.empty:
            return []
        # Pandas/numpy scalar values are normalized before JSON serialization.
        return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))

    def fetch_index_constituents(self, index_code, end_date, lookback_days: int = 30) -> list[str]:
        """Return the constituent stock codes of an index on the latest available date."""
        start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )
        rows = self.fetch_records("index_weight", index_code, start, end_date)
        if not rows:
            return []
        latest = max(str(row["trade_date"]).replace("-", "") for row in rows)
        return sorted(
            {
                str(row["con_code"])
                for row in rows
                if str(row["trade_date"]).replace("-", "") == latest
            }
        )

    def fetch_all_stocks(self) -> list[dict]:
        """Return all listed A-share stocks (ts_code + name) from stock_basic."""
        client = _create_tushare_client(self.token_env)
        frame = client.stock_basic(list_status="L", fields="ts_code,name")
        if frame is None or frame.empty:
            return []
        import json

        return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))
