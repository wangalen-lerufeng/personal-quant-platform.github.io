"""Forward (paper vs market) validation tracker for portfolio research.

Records a Day-0 portfolio (symbols + weights) and a benchmark, then on each
update fetches real Tushare closes to mark both to market, producing a
strategy-vs-benchmark comparison. Read-only: never places orders and never
touches the paper simulation's risk engine.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from .providers import TushareDatasetProvider

_DAILY_DATASET = {"etf": "fund_daily", "stock": "daily", "hk": "hk_daily"}


class ForwardValidationTracker:
    """Persist and advance a single portfolio-vs-benchmark forward test."""

    def __init__(self, store_path: str = "data/forward_validation.json", token_env: str = "TUSHARE_TOKEN"):
        self.store_path = Path(store_path)
        self.token_env = token_env

    def start(
        self,
        symbols,
        weights=None,
        *,
        benchmark_code: str = "000300.SH",
        asset_class: str = "stock",
        predictions: dict | None = None,
    ) -> dict:
        symbols = [str(s).strip() for s in (symbols or []) if str(s).strip()]
        if not symbols:
            raise ValueError("至少需要 1 只标的")
        normalized = self._normalize_weights(symbols, weights)
        provider = TushareDatasetProvider(self.token_env)
        dataset = _DAILY_DATASET.get(asset_class, "daily")
        today = date.today().isoformat()

        entry_prices: dict[str, float] = {}
        entry_dates: dict[str, str] = {}
        for symbol in symbols:
            close, trade_date = self._latest_close(provider, dataset, symbol, today)
            entry_prices[symbol] = close
            entry_dates[symbol] = trade_date
        benchmark_entry, benchmark_date = self._latest_close(
            provider, "index_daily", benchmark_code, today
        )

        state = {
            "run_id": f"fv-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "entry_date": max(entry_dates.values()) if entry_dates else today,
            "benchmark_code": benchmark_code,
            "asset_class": asset_class,
            "symbols": symbols,
            "weights": normalized,
            "entry_prices": entry_prices,
            "latest_prices": dict(entry_prices),
            "benchmark_entry": benchmark_entry,
            "predictions": predictions or {},
            "records": [{
                "date": max([benchmark_date, *entry_dates.values()]) if entry_dates else today,
                "strategy_nav": 1.0,
                "benchmark_nav": 1.0,
                "excess": 0.0,
                "per_symbol_returns": {s: 0.0 for s in symbols},
            }],
        }
        self._save(state)
        return self.report()

    def update(self) -> dict:
        state = self._load()
        if state is None:
            raise ValueError("尚未开始前向验证，请先点击「开始前向验证」")
        provider = TushareDatasetProvider(self.token_env)
        dataset = _DAILY_DATASET.get(state.get("asset_class", "stock"), "daily")
        today = date.today().isoformat()

        latest: dict[str, float] = {}
        latest_dates: dict[str, str] = {}
        for symbol in state["symbols"]:
            close, trade_date = self._latest_close(provider, dataset, symbol, today)
            latest[symbol] = close
            latest_dates[symbol] = trade_date
        benchmark_close, benchmark_date = self._latest_close(
            provider, "index_daily", state["benchmark_code"], today
        )

        trade_date = max([benchmark_date, *latest_dates.values()]) if latest_dates else today
        per_symbol = {
            s: latest[s] / state["entry_prices"][s] - 1.0 for s in state["symbols"]
        }
        strategy_return = sum(state["weights"][s] * per_symbol[s] for s in state["symbols"])
        benchmark_return = benchmark_close / state["benchmark_entry"] - 1.0
        record = {
            "date": trade_date,
            "strategy_nav": round(1.0 + strategy_return, 6),
            "benchmark_nav": round(1.0 + benchmark_return, 6),
            "excess": round(strategy_return - benchmark_return, 6),
            "per_symbol_returns": {s: round(per_symbol[s], 6) for s in state["symbols"]},
        }

        state["records"] = [r for r in state["records"] if r["date"] != trade_date] + [record]
        state["records"].sort(key=lambda r: r["date"])
        state["latest_prices"] = {s: latest[s] for s in state["symbols"]}
        self._save(state)
        return self.report()

    def report(self) -> dict:
        state = self._load()
        if state is None:
            return {"started": False, "research_only": True, "execution_enabled": False}

        records = state.get("records", [])
        strategy_nav = [r["strategy_nav"] for r in records]
        benchmark_nav = [r["benchmark_nav"] for r in records]
        peak = 0.0
        drawdown = []
        for nav in strategy_nav:
            peak = max(peak, nav)
            drawdown.append(round(nav / peak - 1.0, 6) if peak else 0.0)

        per_symbol = records[-1].get("per_symbol_returns", {}) if records else {}
        last_record = records[-1] if records else {}
        strategy_cum = last_record.get("strategy_nav", 1.0) - 1.0
        benchmark_cum = last_record.get("benchmark_nav", 1.0) - 1.0
        excess_cum = last_record.get("excess", 0.0)

        latest_prices = state.get("latest_prices", state.get("entry_prices", {}))
        symbols = state.get("symbols", [])
        weights = state.get("weights", {})
        daily_returns = []
        for i, r in enumerate(records):
            prev_nav = 1.0 if i == 0 else records[i - 1]["strategy_nav"]
            prev_sym = {s: 0.0 for s in symbols} if i == 0 else records[i - 1]["per_symbol_returns"]
            daily_returns.append({
                "date": r["date"],
                "return": round(r["strategy_nav"] / prev_nav - 1.0, 6),
                "per_symbol": {
                    s: round((1.0 + r["per_symbol_returns"].get(s, 0.0)) / (1.0 + prev_sym.get(s, 0.0)) - 1.0, 6)
                    for s in symbols
                },
            })

        return {
            "started": True,
            "run_id": state.get("run_id"),
            "entry_date": state.get("entry_date"),
            "benchmark_code": state.get("benchmark_code"),
            "asset_class": state.get("asset_class"),
            "symbols": symbols,
            "weights": weights,
            "predictions": state.get("predictions", {}),
            "dates": [r["date"] for r in records],
            "strategy_nav": strategy_nav,
            "benchmark_nav": benchmark_nav,
            "excess": [r["excess"] for r in records],
            "drawdown": drawdown,
            "daily_returns": daily_returns,
            "latest_prices": latest_prices,
            "per_symbol": [
                {
                    "symbol": s,
                    "weight": weights.get(s),
                    "entry_price": state.get("entry_prices", {}).get(s),
                    "latest_price": latest_prices.get(s),
                    "return": per_symbol.get(s),
                    "contribution": round((weights.get(s) or 0.0) * (per_symbol.get(s) or 0.0), 6),
                }
                for s in symbols
            ],
            "metrics": {
                "strategy_cumulative_return": round(strategy_cum, 4),
                "benchmark_cumulative_return": round(benchmark_cum, 4),
                "excess_return": round(excess_cum, 4),
                "max_drawdown": round(min(drawdown), 4) if drawdown else 0.0,
                "sample_days": len(records),
            },
            "research_only": True,
            "execution_enabled": False,
        }

    @staticmethod
    def _normalize_weights(symbols, weights) -> dict[str, float]:
        if weights is None:
            return {s: round(1.0 / len(symbols), 6) for s in symbols}
        if isinstance(weights, dict):
            values = {str(s): float(weights.get(s, weights.get(str(s), 0.0))) for s in symbols}
        else:
            values = {s: float(w) for s, w in zip(symbols, weights)}
        total = sum(values.values())
        if total <= 0:
            raise ValueError("权重总和必须大于 0")
        return {s: round(v / total, 6) for s, v in values.items()}

    @staticmethod
    def _latest_close(provider, dataset: str, symbol: str, as_of: str) -> tuple[float, str]:
        start = (date.fromisoformat(as_of) - timedelta(days=30)).isoformat()
        rows = provider.fetch_records(dataset, symbol, start, as_of)
        if not rows:
            raise ValueError(f"未取到 {symbol}（{dataset}）在 {as_of} 前的行情")
        latest = max(rows, key=lambda r: str(r.get("trade_date", "")).replace("-", "")[:8])
        close = latest.get("close")
        if close is None:
            raise ValueError(f"{symbol} 最新行情缺少 close 字段")
        return float(close), ForwardValidationTracker._fmt_date(latest.get("trade_date", as_of))

    @staticmethod
    def _fmt_date(raw) -> str:
        raw = str(raw).strip().replace("-", "")
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) == 8 else raw

    def _load(self) -> dict | None:
        if not self.store_path.exists():
            return None
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _save(self, state: dict) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.store_path)
