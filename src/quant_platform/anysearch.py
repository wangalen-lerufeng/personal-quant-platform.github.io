"""AnySearch integration: convert a keyword into A-share symbols.

Uses AnySearch's structured search (POST https://api.anysearch.com/v1/search) and
extracts A-share stock/ETF codes from the returned content. Read-only research.
"""

import os
import re

import requests

BASE_URL = "https://api.anysearch.com"

_SUFFIXED_RE = re.compile(r"(\d{6})\.(SH|SZ)", re.IGNORECASE)
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _classify(code: str) -> str | None:
    """Return the Qlib-style symbol for a 6-digit code, or None if not A-share."""
    if len(code) != 6 or not code.isdigit():
        return None
    if code[0] in "569":
        return f"{code}.SH"
    if code[0] in "023" or code.startswith(("15", "16")):
        return f"{code}.SZ"
    return None


def keyword_to_symbols(
    keyword: str,
    *,
    token_env: str = "ANYSEARCH_TOKEN",
    max_results: int = 10,
    zone: str = "cn",
    language: str = "zh-CN",
    timeout: float = 30.0,
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
        "query": keyword,
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
    symbols: list[str] = []
    seen: set[str] = set()
    for item in results:
        text = " ".join(str(item.get(key) or "") for key in ("title", "snippet", "content"))
        for match in _SUFFIXED_RE.finditer(text):
            symbol = f"{match.group(1)}.{match.group(2).upper()}"
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
        for match in _CODE_RE.finditer(text):
            symbol = _classify(match.group(1))
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols
