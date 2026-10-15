"""AnySearch integration: convert a keyword into A-share symbols.

Uses AnySearch's structured search (POST https://api.anysearch.com/v1/search) to
find content that mentions related listed companies, then maps the mentioned
company names (and any raw 6-digit codes) back to A-share symbols via Tushare's
stock_basic. Read-only research.
"""

import os
import re

import requests

BASE_URL = "https://api.anysearch.com"

_SUFFIXED_RE = re.compile(r"(\d{6})\.(SH|SZ)", re.IGNORECASE)
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

_stock_cache: tuple[dict[str, str], set[str]] | None = None


def _classify(code: str) -> str | None:
    """Return the Qlib-style symbol for a 6-digit code, or None if not A-share."""
    if len(code) != 6 or not code.isdigit():
        return None
    if code[0] in "569":
        return f"{code}.SH"
    if code[0] in "023" or code.startswith(("15", "16")):
        return f"{code}.SZ"
    return None


def _stock_lookup() -> tuple[dict[str, str], set[str]]:
    """Return ({name: ts_code}, set(ts_code)) for all listed A-share stocks."""
    global _stock_cache
    if _stock_cache is not None:
        return _stock_cache
    from .providers import TushareDatasetProvider

    rows = TushareDatasetProvider("TUSHARE_TOKEN").fetch_all_stocks()
    name_map: dict[str, str] = {}
    code_set: set[str] = set()
    for row in rows:
        name = str(row.get("name") or "").strip()
        code = str(row.get("ts_code") or "").strip()
        if name:
            name_map[name] = code
        if code:
            code_set.add(code)
    _stock_cache = (name_map, code_set)
    return _stock_cache


def keyword_to_symbols(
    keyword: str,
    *,
    token_env: str = "ANYSEARCH_TOKEN",
    max_results: int = 10,
    zone: str = "cn",
    language: str = "zh-CN",
    timeout: float = 30.0,
    limit: int = 20,
) -> list[str]:
    """Search AnySearch for a keyword and return deduplicated A-share symbols."""
    keyword = str(keyword or "").strip()
    if not keyword:
        raise ValueError("keyword is required")

    token = (os.getenv(token_env) or "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "query": f"{keyword} 相关A股上市公司 龙头股",
        "max_results": max(1, min(int(max_results), 10)),
        "zone": zone,
        "language": language,
    }
    try:
        response = requests.post(f"{BASE_URL}/v1/search", json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"AnySearch 请求失败: {error}") from error
    except ValueError as error:
        raise RuntimeError(f"AnySearch 响应解析失败: {error}") from error

    if body.get("code") != 0:
        raise RuntimeError(f"AnySearch 返回错误: {body.get('message')}")

    results = (body.get("data") or {}).get("results", [])
    combined = " ".join(
        str(item.get(key) or "") for item in results for key in ("title", "snippet", "content")
    )

    name_map: dict[str, str] = {}
    code_set: set[str] | None = None
    try:
        name_map, code_set = _stock_lookup()
    except Exception:
        code_set = None

    symbols: list[str] = []
    seen: set[str] = set()

    def add(symbol: str) -> None:
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    # 1. raw 6-digit codes (filtered to real A-share codes when possible)
    for match in _SUFFIXED_RE.finditer(combined):
        symbol = f"{match.group(1)}.{match.group(2).upper()}"
        if code_set is None or symbol in code_set:
            add(symbol)
    for match in _CODE_RE.finditer(combined):
        symbol = _classify(match.group(1))
        if symbol and (code_set is None or symbol in code_set):
            add(symbol)

    # 2. company names -> codes (longest names first for specificity)
    for name in sorted(name_map, key=len, reverse=True):
        if name in combined:
            add(name_map[name])
        if len(symbols) >= limit:
            break

    return symbols[:limit]
