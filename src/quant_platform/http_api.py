import json
import hmac
import ipaddress
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from .adapters import Mt5PaperGateway
from .backtest import BacktestEngine, FuturesRolloverRule, MarketExecutionPolicy
from .config import RuntimeConfig
from .domain import AssetClass, MarketEvent, OrderIntent, OrderSide
from .factor_analysis import analyze_factor, analyze_factor_panel
from .forward_validation import ForwardValidationTracker
from .fundamental_analysis import analyze_fundamental_datasets
from .health import health_snapshot
from .llm_client import OpenAICompatibleChatClient
from .live_readiness import LiveReadinessEvidence, LiveReadinessGate
from .market_data import InMemoryMarketDataFeed
from .market_calendar import MarketCalendar
from .market_state import classify_market_state
from .order_features import analyze_order_flow
from .phase9_service import run_phase9_alpha158
from .pipeline import TargetPosition, plan_orders, submit_to_simulation
from .portfolio_report import build_account_report
from .providers import AkshareForexProvider, TushareDatasetProvider, TushareProvider, TushareStockDailyProvider
from .risk import RiskEngine, RiskLimits, RiskMonitor
from .run_history import get_store
from .runtime import StrategyRuntime
from .simulation import PaperSimulationRegistry
from .simulation_acceptance import SimulationAcceptanceGate, reconcile_simulation_snapshot
from .storage import SqliteAuditStore
from .strategy import MomentumStrategy
from .trading import TradingPipeline
from .tushare_credentials import save_tushare_token, tushare_token_status


def backtest_payload(payload: dict) -> dict:
    """Run a research-only momentum backtest from structured event data."""
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("events must be a non-empty list")
    events = [
        MarketEvent(
            symbol=row["symbol"],
            asset_class=AssetClass(row["asset_class"]),
            price=float(row["price"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
        for row in raw_events
    ]
    execution_policy = None
    if bool(payload.get("enforce_market_rules", False)):
        holidays = {
            date.fromisoformat(str(value)) for value in payload.get("holidays", [])
        }
        rollover_rules = [
            FuturesRolloverRule(
                expiring_symbol=str(item["expiring_symbol"]),
                successor_symbol=str(item["successor_symbol"]),
                effective_at=datetime.fromisoformat(str(item["effective_at"])),
            )
            for item in payload.get("rollover_rules", [])
        ]
        execution_policy = MarketExecutionPolicy(
            calendar=MarketCalendar(holidays),
            previous_closes={
                str(symbol): float(value)
                for symbol, value in payload.get("previous_closes", {}).items()
            },
            price_limit_percentages={
                str(symbol): float(value)
                for symbol, value in payload.get("price_limit_percentages", {}).items()
            },
            rollover_rules=rollover_rules,
            auto_rollover=bool(payload.get("auto_rollover", False)),
        )
    result = BacktestEngine(
        MomentumStrategy(lookback=int(payload.get("lookback", 2)), quantity=float(payload.get("quantity", 1))),
        commission_rate=float(payload.get("commission_rate", 0)),
        slippage_rate=float(payload.get("slippage_rate", 0)),
        execution_policy=execution_policy,
    ).run(events)
    return {**result.to_dict(), "execution_enabled": False, "research_only": True}


def order_features_payload(payload: dict) -> dict:
    trades = payload.get("trades", [])
    if not isinstance(trades, list) or not trades:
        raise ValueError("trades must be a non-empty list")
    result = analyze_order_flow(trades, float(payload.get("large_trade_threshold", 0.0)))
    return {**result, "execution_enabled": False, "research_only": True}


def simulation_risk_payload(payload: dict) -> dict:
    """Replay paper-account equity through conservative risk thresholds."""
    equity_series = payload.get("equity_series", [])
    if not isinstance(equity_series, list) or not equity_series:
        raise ValueError("equity_series must be a non-empty list")
    initial_equity = float(payload.get("initial_equity", equity_series[0]))
    limits = RiskLimits(
        kill_switch=bool(payload.get("kill_switch", False)),
        max_daily_loss=(
            None if payload.get("max_daily_loss") is None else float(payload["max_daily_loss"])
        ),
        max_drawdown=(
            None if payload.get("max_drawdown") is None else float(payload["max_drawdown"])
        ),
    )
    monitor = RiskMonitor(limits, initial_equity)
    history = [monitor.observe(float(equity)) for equity in equity_series]
    return {
        "initial_equity": initial_equity,
        "latest": history[-1],
        "history": history,
        "research_only": True,
        "execution_enabled": False,
    }


def factor_analysis_payload(payload: dict) -> dict:
    """Analyze a factor/forward-return sample without mutating strategy state."""
    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    factor_column = str(payload.get("factor_column", "factor"))
    return_column = str(payload.get("return_column", "forward_return"))
    result = analyze_factor(
        pd.DataFrame.from_records(records),
        factor_column,
        return_column,
        groups=int(payload.get("groups", 5)),
    )
    return {**result, "research_only": True, "execution_enabled": False}


def factor_panel_analysis_payload(payload: dict) -> dict:
    """Run date-by-asset factor diagnostics against explicit forward labels."""
    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    return_columns = payload.get("return_columns", ["forward_return"])
    if not isinstance(return_columns, list):
        raise ValueError("return_columns must be a list")
    result = analyze_factor_panel(
        pd.DataFrame.from_records(records),
        factor_column=str(payload.get("factor_column", "factor")),
        return_columns=return_columns,
        groups=int(payload.get("groups", 5)),
        date_column=str(payload.get("date_column", "timestamp")),
        asset_column=str(payload.get("asset_column", "symbol")),
        out_of_sample_start=payload.get("out_of_sample_start"),
        dollar_volume_column=payload.get("dollar_volume_column"),
        participation_rate=float(payload.get("participation_rate", 0.01)),
    )
    return {**result, "research_only": True, "execution_enabled": False}


def market_state_payload(payload: dict) -> dict:
    """Classify an explainable market regime from supplied historical bars."""
    bars = payload.get("bars", [])
    if not isinstance(bars, list) or not bars:
        raise ValueError("bars must be a non-empty list")
    result = classify_market_state(
        pd.DataFrame.from_records(bars),
        window=int(payload.get("window", 20)),
    )
    return {**result, "research_only": True, "execution_enabled": False}


def live_readiness_payload(payload: dict) -> dict:
    boolean_fields = (
        "out_of_sample_passed",
        "reconciliation_passed",
        "risk_limits_configured",
    )
    for field in boolean_fields:
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    evidence = LiveReadinessEvidence(
        out_of_sample_passed=payload["out_of_sample_passed"],
        paper_trading_days=int(payload.get("paper_trading_days", 0)),
        reconciliation_passed=payload["reconciliation_passed"],
        risk_limits_configured=payload["risk_limits_configured"],
        manual_approval_id=payload.get("manual_approval_id"),
    )
    return LiveReadinessGate(
        minimum_paper_days=int(payload.get("minimum_paper_days", 30))
    ).assess(evidence)


def phase9_payload(payload: dict, control_plane, simulation_registry, config=None) -> dict:
    """Run the Phase 9 pipeline, risk-gate, submit to paper, and add an LLM analysis."""
    symbols = payload.get("symbols", [])
    top_k = int(payload.get("top_k", 3))
    start_date = str(payload.get("start_date", "")).strip()
    end_date = str(payload.get("end_date", "")).strip() or date.today().isoformat()
    asset_class = str(payload.get("asset_class", "etf")).strip().lower() or "etf"
    if not start_date:
        raise ValueError("start_date is required")

    index_code = str(payload.get("index_code", "")).strip()
    if index_code and index_code != "manual":
        symbols = TushareDatasetProvider("TUSHARE_TOKEN").fetch_index_constituents(index_code, end_date)
        if len(symbols) < 2:
            raise ValueError(f"指数 {index_code} 未取到成分股，请确认指数代码与日期")

    if not isinstance(symbols, list) or not symbols:
        raise ValueError("symbols must be a non-empty list")

    feature_expressions = payload.get("feature_expressions", [])
    if isinstance(feature_expressions, str):
        feature_expressions = [
            item.strip() for item in feature_expressions.replace("\n", ",").split(",") if item.strip()
        ]

    research = run_phase9_alpha158(
        symbols, start_date, end_date, top_k=top_k, asset_class=asset_class,
        feature_expressions=feature_expressions or None,
    )
    risk_engine = control_plane.risk_engine if control_plane is not None else RiskEngine(RiskLimits())
    targets = [
        TargetPosition(symbol=item["symbol"], weight=item["weight"])
        for item in research["targets"]
    ]
    plan = plan_orders({}, targets, risk_engine)

    run_id = f"phase9-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    account_id = run_id
    session = simulation_registry.create(account_id, initial_cash=1_000_000.0)
    submission = submit_to_simulation(plan, session)

    result = {
        **research,
        "run_id": run_id,
        "account_id": account_id,
        "orders": [{"symbol": order.symbol, "side": order.side.value, "quantity": order.quantity} for order in plan.orders],
        "rejected": [dict(item) for item in plan.rejected],
        "audit": [{"stage": entry.stage, "symbol": entry.symbol, "detail": dict(entry.detail)} for entry in plan.audit],
        "submission": submission,
        "paper_only": True,
        "execution_enabled": False,
    }
    _record_phase9_run(
        symbols, top_k, asset_class, index_code, result, run_id=run_id,
        feature_expressions=feature_expressions, start_date=start_date, end_date=end_date,
    )
    _persist_last_phase9(research, asset_class, index_code, run_id, account_id)
    result["llm_analysis"] = (
        _phase9_llm_analysis(result, config)
        if config is not None and config.ai_model_enabled
        else None
    )
    return result


def _record_phase9_run(
    symbols, top_k, asset_class, index_code, result,
    run_history=None, run_id=None, feature_expressions=None, start_date=None, end_date=None,
) -> str:
    store = run_history or get_store()
    record = {
        "kind": "phase9",
        "asset_class": asset_class,
        "index_code": index_code or "manual",
        "top_k": top_k,
        "symbols": list(symbols),
        "feature_expressions": list(feature_expressions) if feature_expressions else [],
        "start_date": start_date,
        "end_date": end_date,
        "candidates": result.get("candidates", []),
        "targets": result.get("targets", []),
        "baseline_rank_ic": result.get("baseline", {}).get("rank_ic"),
        "ensemble_rank_ic": result.get("ensemble", {}).get("rank_ic"),
        "orders": result.get("orders", []),
        "rejected_count": len(result.get("rejected", [])),
        "account_id": result.get("account_id"),
        "research_only": True,
        "execution_enabled": False,
    }
    if run_id:
        record["run_id"] = run_id
    return store.add(record)


def _persist_last_phase9(research: dict, asset_class: str, index_code: str, phase9_run_id: str, account_id: str) -> None:
    targets = research.get("targets", [])
    symbols = [str(item["symbol"]) for item in targets]
    weights = {str(item["symbol"]): float(item["weight"]) for item in targets}
    benchmark_code = index_code if index_code and index_code != "manual" else "000300.SH"
    path = Path("data/last_phase9.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase9_run_id": phase9_run_id,
                "account_id": account_id,
                "symbols": symbols,
                "weights": weights,
                "asset_class": asset_class,
                "benchmark_code": benchmark_code,
                "candidates": research.get("candidates", []),
                "rank_ic": research.get("baseline", {}).get("rank_ic"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _build_phase9_analysis_prompt(result: dict) -> str:
    panel = result.get("panel", {})
    baseline = result.get("baseline", {})
    ensemble = result.get("ensemble", {})
    candidates = result.get("candidates", [])
    verification = result.get("verification", [])
    sizing = result.get("sizing", {})
    targets = result.get("targets", [])

    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [
        "你是一名量化投资研究助手。以下是平台自动跑出的「组合研究」流水线结果，请用中文做一份客观、结构化的投资解读。",
        "注意：这些是研究/纸面结果，不构成投资建议；只基于下面给出的数据解读，不要编造任何数字。",
        "",
        f"1. 数据面板：{panel.get('rows', '?')} 行 × {panel.get('symbols', '?')} 只，区间 {panel.get('start', '?')} ~ {panel.get('end', '?')}。",
        f"2. 候选池模型样本外表现：LightGBM Rank IC={fmt(baseline.get('rank_ic'))} ICIR={fmt(baseline.get('icir'))}；DoubleEnsemble Rank IC={fmt(ensemble.get('rank_ic'))} ICIR={fmt(ensemble.get('icir'))}。",
        f"3. 最新候选池：{'、'.join(candidates) if candidates else '无'}。",
    ]
    for item in verification:
        tags = "、".join(item.get("risk_tags", [])) or "无"
        lines.append(f"   - {item.get('symbol')}：验真 {item.get('verdict')}（风险标签：{tags}）")
    target_str = "无"
    if targets:
        target_str = "、".join(f"{t['symbol']}:{t['weight']:.4f}" for t in targets)
    lines.append(f"4. PPO 调整因子：{fmt(sizing.get('adjustment'))}。")
    lines.append(f"5. 目标仓位：{target_str}。")
    lines.append(
        f"6. 风控订单：{len(result.get('orders', []))} 笔、拒绝 {len(result.get('rejected', []))} 笔；"
        f"纸面提交 {result.get('submission', {}).get('order_count', 0)} 笔。"
    )
    lines.append("")
    lines.append("请分三段输出：A. 组合整体解读；B. 各候选标的简要点评；C. 风险提示与局限（尤其 IC 稳定性、样本规模、真实执行已关闭）。")
    return "\n".join(lines)


def _phase9_llm_analysis(result: dict, config) -> str | None:
    try:
        prompt = _build_phase9_analysis_prompt(result)
        client = OpenAICompatibleChatClient(
            config.ai_base_url,
            config.ai_model,
            config.ai_api_key_env,
            config.ai_temperature,
            config.ai_max_tokens,
            config.ai_timeout_seconds,
        )
        return client.complete(prompt)
    except Exception as error:
        return f"[大模型解读失败] {error}"


def factor_symbol_payload(payload: dict, config=None) -> dict:
    """Compute factor time series + stats for a single symbol, plus an LLM analysis."""
    from .qlib_alpha158 import analyze_single_symbol_factors

    symbol = str(payload.get("symbol", "")).strip()
    factor_expressions = payload.get("factor_expressions", [])
    if isinstance(factor_expressions, str):
        factor_expressions = [
            item.strip() for item in factor_expressions.replace("\n", ",").split(",") if item.strip()
        ]
    start_date = str(payload.get("start_date", "")).strip()
    end_date = str(payload.get("end_date", "")).strip() or date.today().isoformat()
    asset_class = str(payload.get("asset_class", "etf")).strip().lower() or "etf"
    if not symbol:
        raise ValueError("symbol is required")
    if not start_date:
        raise ValueError("start_date is required")

    result = analyze_single_symbol_factors(
        symbol, factor_expressions, start_date, end_date, asset_class=asset_class,
    )
    result["llm_analysis"] = (
        _factor_llm_analysis(result, config)
        if config is not None and config.ai_model_enabled
        else None
    )
    return result


def _build_factor_analysis_prompt(result: dict) -> str:
    lines = [
        f"你是量化投资研究助手。以下是股票 {result['symbol']} 的因子时序分析结果，请结合这些因子做客观解读。",
        "注意：这是研究/统计结果，不构成投资建议；只基于数据解读，不要编造任何数字。",
        "",
        f"数据区间：{result['start']} ~ {result['end']}。",
    ]
    for factor in result["factors"]:
        percentile = factor["percentile"]
        percentile_str = "—" if percentile is None else f"{percentile * 100:.1f}%"
        ic = "—" if factor["ic"] is None else f"{factor['ic']:.4f}"
        lines.append(
            f"- {factor['expression']}：当前值={factor['latest']:.4f}，历史分位={percentile_str}，"
            f"与次日收益相关性 IC={ic}，均值={factor['mean']:.4f}，标准差={factor['std']:.4f}"
        )
    lines.append("")
    lines.append(
        "请输出：A. 该股当前趋势/位置的客观判断；B. 哪些因子信号最强（看 IC 绝对值）；"
        "C. 风险提示与局限（单股因子 IC 噪声大、非确定性预测）。"
    )
    return "\n".join(lines)


def _factor_llm_analysis(result: dict, config) -> str | None:
    try:
        prompt = _build_factor_analysis_prompt(result)
        client = OpenAICompatibleChatClient(
            config.ai_base_url,
            config.ai_model,
            config.ai_api_key_env,
            config.ai_temperature,
            config.ai_max_tokens,
            config.ai_timeout_seconds,
        )
        return client.complete(prompt)
    except Exception as error:
        return f"[大模型解读失败] {error}"


def forward_validation_payload(payload: dict, tracker: ForwardValidationTracker, run_history=None) -> dict:
    """Start / update / report a portfolio-vs-benchmark forward validation."""
    store = run_history or get_store()
    action = str(payload.get("action", "report")).strip().lower()
    if action == "start":
        symbols = payload.get("symbols", [])
        weights = payload.get("weights")
        benchmark_code = str(payload.get("benchmark_code", "000300.SH")).strip() or "000300.SH"
        asset_class = str(payload.get("asset_class", "stock")).strip().lower() or "stock"
        predictions = payload.get("predictions")
        report = tracker.start(
            symbols, weights, benchmark_code=benchmark_code,
            asset_class=asset_class, predictions=predictions,
        )
        _record_forward_start(report, store)
        return report
    if action == "start_last":
        report = _start_forward_validation_from_last(tracker)
        _record_forward_start(report, store, source=report.get("source_phase9_run_id"))
        return report
    if action == "update":
        report = tracker.update()
        _record_forward_update(report, store)
        return report
    return tracker.report()


def _record_forward_start(report: dict, store, source: str | None = None) -> None:
    store.add({
        "kind": "forward",
        "run_id": report.get("run_id"),
        "source_phase9_run_id": source,
        "benchmark_code": report.get("benchmark_code"),
        "asset_class": report.get("asset_class"),
        "entry_date": report.get("entry_date"),
        "symbols": report.get("symbols"),
        "weights": report.get("weights"),
        "metrics": report.get("metrics"),
        "per_symbol": report.get("per_symbol"),
        "research_only": True,
        "execution_enabled": False,
    })


def _record_forward_update(report: dict, store) -> None:
    store.update(report.get("run_id"), {
        "metrics": report.get("metrics"),
        "per_symbol": report.get("per_symbol"),
    })


def _start_forward_validation_from_last(tracker: ForwardValidationTracker) -> dict:
    path = Path("data/last_phase9.json")
    if not path.exists():
        raise ValueError("还没有运行过组合研究，请先在组合研究页跑一次流水线")
    data = json.loads(path.read_text(encoding="utf-8"))
    symbols = data.get("symbols", [])
    if not symbols:
        raise ValueError("最近一次组合研究没有产生目标仓位")
    report = tracker.start(
        symbols,
        data.get("weights"),
        benchmark_code=str(data.get("benchmark_code", "000300.SH")),
        asset_class=str(data.get("asset_class", "stock")),
        predictions={
            "candidates": data.get("candidates", []),
            "rank_ic": data.get("rank_ic"),
        },
    )
    report["source_phase9_run_id"] = data.get("phase9_run_id")
    return report


def run_history_delete_payload(payload: dict, simulation_registry=None) -> dict:
    run_id = str(payload.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run_id is required")
    store = get_store()
    record = store.get(run_id)
    if record is None:
        raise ValueError(f"未知的运行记录: {run_id}")
    account_id = record.get("account_id")
    result = store.delete(run_id)
    deleted_account = None
    if account_id and simulation_registry is not None:
        try:
            simulation_registry.delete(str(account_id))
            deleted_account = str(account_id)
        except KeyError:
            pass
    result.update({
        "deleted_account": deleted_account,
        "paper_only": True,
        "execution_enabled": False,
    })
    return result


def simulation_create_payload(payload: dict, registry: PaperSimulationRegistry) -> dict:
    account_id = str(payload.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("account_id is required")
    limits = RiskLimits(
        kill_switch=bool(payload.get("kill_switch", False)),
        max_order_quantity=float(payload.get("max_order_quantity", 1_000_000)),
        max_position_quantity=(
            None
            if payload.get("max_position_quantity") is None
            else float(payload["max_position_quantity"])
        ),
        max_order_value=(
            None if payload.get("max_order_value") is None else float(payload["max_order_value"])
        ),
        max_daily_loss=(
            None if payload.get("max_daily_loss") is None else float(payload["max_daily_loss"])
        ),
        max_drawdown=(
            None if payload.get("max_drawdown") is None else float(payload["max_drawdown"])
        ),
    )
    session = registry.create(
        account_id=account_id,
        initial_cash=float(payload.get("initial_cash", 0)),
        risk_limits=limits,
        max_consecutive_losses=(
            None
            if payload.get("max_consecutive_losses") is None
            else int(payload["max_consecutive_losses"])
        ),
        market_data_timeout_seconds=(
            None
            if payload.get("market_data_timeout_seconds") is None
            else float(payload["market_data_timeout_seconds"])
        ),
        max_api_failures=(
            None if payload.get("max_api_failures") is None else int(payload["max_api_failures"])
        ),
        latency_events=int(payload.get("latency_events", 0)),
        max_participation_rate=float(payload.get("max_participation_rate", 0.1)),
        slippage_rate=float(payload.get("slippage_rate", 0.0)),
    )
    return session.state()


def simulation_order_payload(payload: dict, registry: PaperSimulationRegistry) -> dict:
    account_id = str(payload.get("account_id", "")).strip()
    session = registry.get(account_id)
    raw_asset_class = payload.get("asset_class")
    intent = OrderIntent(
        symbol=str(payload["symbol"]),
        side=OrderSide(str(payload["side"]).lower()),
        quantity=float(payload["quantity"]),
        strategy_id=str(payload.get("strategy_id", "paper-console")),
        asset_class=None if raw_asset_class is None else AssetClass(str(raw_asset_class)),
        client_order_id=payload.get("client_order_id"),
        reference_price=(
            None if payload.get("reference_price") is None else float(payload["reference_price"])
        ),
    )
    order_id = session.submit(intent)
    return {
        "account_id": account_id,
        "order_id": order_id,
        "status": "accepted",
        "paper_only": True,
        "execution_enabled": False,
    }


def simulation_market_payload(payload: dict, registry: PaperSimulationRegistry) -> dict:
    session = registry.get(str(payload.get("account_id", "")).strip())
    return session.on_market(
        symbol=str(payload["symbol"]),
        price=float(payload["price"]),
        volume=float(payload["volume"]),
    )


def simulation_control_payload(payload: dict, registry: PaperSimulationRegistry) -> dict:
    session = registry.get(str(payload.get("account_id", "")).strip())
    action = str(payload.get("action", "")).strip()
    if action == "record_trade_pnl":
        session.record_trade_pnl(float(payload["pnl"]))
    elif action == "record_api_result":
        success = payload.get("success")
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        session.record_api_result(success)
    elif action == "kill_switch":
        session.activate_kill_switch(str(payload.get("reason", "operator_requested")))
    else:
        raise ValueError(f"unsupported simulation control action: {action}")
    return session.state()


def simulation_delete_payload(payload: dict, registry: PaperSimulationRegistry) -> dict:
    account_id = str(payload.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("account_id is required")
    return registry.delete(account_id)


def simulation_account_report_payload(payload: dict, registry: PaperSimulationRegistry) -> dict:
    account_id = str(payload.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("account_id is required")
    return build_account_report(registry.get(account_id))


def simulation_reconciliation_payload(
    payload: dict,
    registry: PaperSimulationRegistry,
) -> dict:
    account_id = str(payload.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("account_id is required")
    session = registry.get(account_id)
    return reconcile_simulation_snapshot(
        session.snapshot(),
        tolerance=float(payload.get("tolerance", 1e-8)),
    )


def simulation_acceptance_payload(
    payload: dict,
    registry: PaperSimulationRegistry,
) -> dict:
    account_id = str(payload.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("account_id is required")
    session = registry.get(account_id)
    account_snapshots = []
    if registry.account_store is not None and hasattr(
        registry.account_store, "list_snapshots"
    ):
        account_snapshots = registry.account_store.list_snapshots(account_id)
    return SimulationAcceptanceGate(
        minimum_trading_days=int(payload.get("minimum_trading_days", 30))
    ).assess(session.snapshot(), account_snapshots)


def paper_demo_payload() -> dict:
    feed = InMemoryMarketDataFeed()
    store = SqliteAuditStore(":memory:")
    runtime = StrategyRuntime(
        feed,
        MomentumStrategy(lookback=2, quantity=0.01),
        TradingPipeline(RiskEngine(RiskLimits()), Mt5PaperGateway(), audit_store=store),
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for offset, price in enumerate((1.0800, 1.0810, 1.0830, 1.0850)):
        feed.publish(MarketEvent("EURUSD", AssetClass.FX, price, start + timedelta(days=offset)))
    return {"paper_results": len(runtime.results), "orders": store.list_orders(), "execution_enabled": False, "paper_only": True}


def research_agent_payload(prompt: str, research_tools=None) -> dict:
    """Small deterministic orchestration shell, intentionally read-only."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    steps = ["读取 system_status", "读取 risk_status"]
    answer = "已完成安全状态检查；当前系统为 paper-only，AI 不具备执行权限。"
    lowered = prompt.lower()
    if any(word in lowered for word in ("因子", "ic", "rank ic")):
        if any(word in lowered for word in ("icir", "衰减", "容量", "自相关", "样本外", "换手率")):
            steps += ["读取按日横截面因子样本", "运行 analyze_factor_panel", "汇总 IC 序列、ICIR、衰减、换手率、容量和样本外表现"]
            answer += " 检测到高级因子诊断意图，应提交带时间、资产和明确未来收益标签的面板数据。"
        else:
            steps += ["读取因子样本", "运行 analyze_factor", "汇总 IC、Rank IC 和分组收益"]
            answer += " 检测到因子研究意图，应提交因子值与下一期收益运行只读分析。"
    elif any(word in lowered for word in ("市场状态", "趋势", "震荡", "波动状态")):
        steps += ["读取历史 K 线", "运行 classify_market_state", "汇总状态与识别依据"]
        answer += " 检测到市场状态意图，应使用历史 K 线运行可解释状态识别。"
    elif any(word in lowered for word in ("订单流", "成交不平衡", "vwap", "大单")):
        steps += ["读取逐笔成交", "运行 analyze_order_features", "汇总订单流特征"]
        answer += " 检测到订单流研究意图，应使用标准化逐笔成交运行只读分析。"
    elif (
        any(word in lowered for word in ("风险", "回撤", "熔断", "权益"))
        and not any(word in lowered for word in ("回测", "策略", "backtest"))
    ):
        steps += ["读取仿真权益", "运行 monitor_simulation_risk", "检查亏损、回撤与熔断阈值"]
        answer += " 检测到风险监控意图，应回放权益序列并检查保守阈值。"
    elif any(word in lowered for word in ("回测", "策略", "backtest")):
        steps += ["读取 market_data", "运行 run_backtest", "汇总收益、胜率和最大回撤"]
        answer += " 检测到回测意图，下一步应使用历史行情运行研究回测。"
    elif any(word in lowered for word in ("持仓", "组合", "portfolio")):
        steps += ["读取 portfolio_state", "读取 market_data", "生成组合估值快照"]
        answer += " 检测到组合分析意图，已规划组合估值查询。"
    else:
        steps += ["读取 market_data", "读取 experiment_results", "生成研究摘要"]
        answer += " 已规划行情与实验结果查询。"
    return {"prompt": prompt, "orchestration": steps, "answer": answer, "tools_are_read_only": True, "execution_enabled": False}


def model_research_payload(prompt: str, config: RuntimeConfig, research_tools=None) -> dict:
    client = OpenAICompatibleChatClient(config.ai_base_url, config.ai_model, config.ai_api_key_env, config.ai_temperature, config.ai_max_tokens, config.ai_timeout_seconds)
    answer = client.complete(prompt)
    body = {"prompt": prompt, "answer": answer, "provider": config.ai_provider, "model": config.ai_model, "tools_are_read_only": True, "execution_enabled": False, "evidence_based": False, "tool_executed": False}
    model_calls = extract_research_tool_calls(answer)
    planned_calls = plan_stock_research_calls(prompt)
    tool_calls = _merge_research_calls(planned_calls, model_calls)
    if not tool_calls and re.search(r"回测|最大回撤|backtest", prompt, flags=re.IGNORECASE):
        symbol_match = re.search(r"\b(?:[A-Z]{6}|\d{6}\.(?:SH|SZ))\b", prompt)
        end = date.today()
        start = end - timedelta(days=365 * 3)
        symbol = symbol_match.group(0) if symbol_match else "EURUSD"
        provider = "tushare" if re.search(r"tushare|ETF|基金", prompt, flags=re.IGNORECASE) or re.search(r"\d{6}\.(?:SH|SZ)", symbol) else "akshare"
        tool_calls = [{"tool": "run_backtest", "arguments": {"provider": provider, "symbol": symbol, "start_date": start.isoformat(), "end_date": end.isoformat(), "lookback": 20, "quantity": 1}}]
    all_calls = []
    all_results = []
    rounds = 0
    while tool_calls and rounds < 6 and len(all_calls) < 15:
        batch = tool_calls[: 15 - len(all_calls)]
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            results = list(
                executor.map(
                    lambda call: execute_research_tool_call_safely(
                        call, research_tools
                    ),
                    batch,
                )
            )
        all_calls.extend(batch)
        all_results.extend(results)
        analysis = build_fundamental_analysis_result(all_results, date.today())
        model_results = results + ([analysis] if analysis else [])
        answer = client.complete_with_tool_result(
            prompt,
            answer,
            {"results": model_results, "completed_calls": len(all_calls)},
        )
        rounds += 1
        tool_calls = extract_research_tool_calls(answer)
    final_analysis = build_fundamental_analysis_result(all_results, date.today())
    if final_analysis:
        all_results.append(final_analysis)
    if all_calls:
        body["tool_call"] = all_calls[0]
        body["tool_result"] = all_results[0]
        body["tool_calls"] = all_calls
        body["tool_results"] = all_results
        body["answer"] = answer
        body["evidence_based"] = any(
            "error" not in result and bool(result.get("data")) for result in all_results
        )
        body["tool_executed"] = True
    return body


def plan_stock_research_calls(prompt: str) -> list[dict]:
    """Build the mandatory, reproducible data plan for A-share deep research."""
    symbol_match = re.search(r"(?<!\d)\d{6}\.(?:SH|SZ|BJ)(?![A-Z])", prompt, flags=re.IGNORECASE)
    requested = [
        dataset for dataset in (
            "stock_basic", "daily", "daily_basic", "income", "balancesheet", "cashflow", "fina_indicator"
        )
        if dataset in prompt
    ]
    if not symbol_match or not ("深度研究" in prompt or len(requested) >= 3):
        return []
    end_match = re.search(r"研究截止日(?:为|：|:)\s*(\d{4}-\d{2}-\d{2})", prompt)
    end_date = date.fromisoformat(end_match.group(1)) if end_match else date.today()
    start_date = end_date - timedelta(days=365 * 5)
    datasets = requested or [
        "stock_basic", "daily", "daily_basic", "income", "balancesheet", "cashflow", "fina_indicator"
    ]
    return [{
        "tool": "get_market_data",
        "arguments": {
            "provider": "tushare",
            "symbol": symbol_match.group(0).upper(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "dataset": dataset,
            "asset_class": "equity",
        },
    } for dataset in datasets]


def _merge_research_calls(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for call in primary + secondary:
        arguments = call.get("arguments", {})
        key = (call.get("tool"), arguments.get("provider"), arguments.get("symbol"), arguments.get("dataset"))
        if key not in seen:
            seen.add(key)
            merged.append(call)
    return merged


def build_fundamental_analysis_result(results: list[dict], as_of_date: date | None = None) -> dict | None:
    financial_datasets = {"income", "balancesheet", "cashflow", "fina_indicator"}
    if not any(result.get("dataset") in financial_datasets and result.get("data") for result in results):
        return None
    return {
        "tool": "analyze_fundamentals",
        "dataset": "fundamental_analysis",
        "data": analyze_fundamental_datasets(results, as_of_date or date.today()),
        "research_only": True,
        "execution_enabled": False,
    }


def execute_research_tool_call_safely(tool_call: dict, research_tools=None) -> dict:
    try:
        return execute_research_tool_call(tool_call, research_tools)
    except Exception as error:
        safe_error = re.sub(r"(?i)(token=)[^&\s]+", r"\1[redacted]", str(error))
        return {
            "tool": tool_call.get("tool"),
            "dataset": tool_call.get("arguments", {}).get("dataset"),
            "error": safe_error,
            "execution_enabled": False,
        }


def extract_research_tool_calls(answer: str) -> list[dict]:
    matches = re.findall(r"\{\s*\"tool\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\}", answer, flags=re.DOTALL)
    calls = []
    for match in matches:
        try:
            value = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("arguments", {}), dict):
            value["tool"] = _canonical_research_tool_name(value.get("tool"))
            calls.append(value)
    if calls:
        return calls

    marker = r"[|｜]{2}DSML[|｜]{2}"
    invokes = re.findall(
        rf"<{marker}invoke\s+name=\"([^\"]+)\">(.*?)</{marker}invoke>",
        answer,
        flags=re.DOTALL,
    )
    for tool_name, body in invokes:
        arguments = {}
        for name, value in re.findall(
            rf"<{marker}parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</{marker}parameter>",
            body,
            flags=re.DOTALL,
        ):
            arguments[name] = value.strip()
        calls.append({"tool": _canonical_research_tool_name(tool_name), "arguments": arguments})
    return calls


def _canonical_research_tool_name(tool: str | None) -> str | None:
    aliases = {"get_fundamental_data": "get_market_data"}
    return aliases.get(tool, tool)


def extract_research_tool_call(answer: str) -> dict | None:
    calls = extract_research_tool_calls(answer)
    return calls[0] if calls else None


def execute_research_tool_call(tool_call: dict, research_tools=None) -> dict:
    allowed = {
        "get_market_data",
        "run_backtest",
        "portfolio_state",
        "risk_status",
        "analyze_order_features",
        "analyze_factor",
        "analyze_factor_panel",
        "classify_market_state",
        "monitor_simulation_risk",
    }
    tool = _canonical_research_tool_name(tool_call.get("tool"))
    if tool not in allowed:
        return {"error": f"tool not allowed: {tool}", "execution_enabled": False}
    arguments = tool_call.get("arguments", {})
    if "token_env" in arguments:
        return {
            "tool": tool,
            "error": "credential selector arguments are not allowed",
            "execution_enabled": False,
        }
    local_analytics = {
        "analyze_order_features": order_features_payload,
        "analyze_factor": factor_analysis_payload,
        "analyze_factor_panel": factor_panel_analysis_payload,
        "classify_market_state": market_state_payload,
        "monitor_simulation_risk": simulation_risk_payload,
    }
    if tool in local_analytics:
        return {
            "tool": tool,
            "data": local_analytics[tool](arguments),
            "execution_enabled": False,
            "research_only": True,
        }
    if research_tools is None and tool != "run_backtest":
        return {"error": "research tools are unavailable", "execution_enabled": False}
    provider = arguments.get("provider", "akshare").lower()
    if provider == "tushare":
        dataset = str(arguments.get("dataset", "fund_daily")).lower()
        asset_class = str(arguments.get("asset_class", "etf")).lower()
        symbol = arguments.get("symbol", "")
        if not arguments.get("start_date") or not arguments.get("end_date"):
            return {"tool": tool, "error": "Tushare requires start_date and end_date", "execution_enabled": False}
        if dataset not in {"daily", "fund_daily"}:
            if tool != "get_market_data":
                return {"tool": tool, "error": f"dataset {dataset} is not price data for backtesting", "execution_enabled": False}
            try:
                records = TushareDatasetProvider("TUSHARE_TOKEN").fetch_records(
                    dataset, symbol, arguments["start_date"], arguments["end_date"]
                )
            except ValueError as error:
                return {"tool": tool, "provider": "tushare", "dataset": dataset, "error": str(error), "execution_enabled": False}
            return {"tool": tool, "provider": "tushare", "dataset": dataset, "data": records, "execution_enabled": False, "research_only": True}
        provider_client = (
            TushareStockDailyProvider("TUSHARE_TOKEN")
            if dataset == "daily" or asset_class == "equity"
            else TushareProvider("TUSHARE_TOKEN")
        )
        events = provider_client.fetch(symbol, arguments["start_date"], arguments["end_date"])
        data = [{"symbol": event.symbol, "asset_class": event.asset_class.value, "price": event.price, "timestamp": event.timestamp.isoformat()} for event in events]
        if tool == "get_market_data":
            return {"tool": tool, "provider": "tushare", "dataset": dataset, "data": data, "execution_enabled": False}
        if not data:
            return {"tool": tool, "provider": "tushare", "error": "no historical market data available", "execution_enabled": False}
        result = BacktestEngine(MomentumStrategy(int(arguments.get("lookback", 20)), float(arguments.get("quantity", 1)), exit_on_negative=True)).run(events)
        return {"tool": tool, "provider": "tushare", "data": result.to_dict(), "execution_enabled": False, "research_only": True}
    if tool == "get_market_data":
        data = research_tools.market_data(arguments.get("symbol"), 100)
        if not data and arguments.get("provider", "akshare") == "akshare" and arguments.get("start_date") and arguments.get("end_date"):
            events = AkshareForexProvider().fetch(arguments.get("symbol", "EURUSD"), arguments["start_date"], arguments["end_date"])
            data = [{"symbol": event.symbol, "asset_class": event.asset_class.value, "price": event.price, "timestamp": event.timestamp.isoformat()} for event in events]
        return {"tool": tool, "data": data, "execution_enabled": False}
    if tool == "run_backtest":
        if arguments.get("provider", "akshare") != "akshare" or not arguments.get("start_date") or not arguments.get("end_date"):
            return {"tool": tool, "error": "no historical market data available", "execution_enabled": False}
        events = AkshareForexProvider().fetch(arguments.get("symbol", "EURUSD"), arguments["start_date"], arguments["end_date"])
        if not events:
            return {"tool": tool, "error": "no historical market data available", "execution_enabled": False}
        result = BacktestEngine(MomentumStrategy(int(arguments.get("lookback", 20)), float(arguments.get("quantity", 1)), exit_on_negative=True)).run(events)
        return {"tool": tool, "data": result.to_dict(), "execution_enabled": False, "research_only": True}
    if tool == "portfolio_state":
        return {"tool": tool, "data": research_tools.portfolio_state(), "execution_enabled": False}
    return {"tool": tool, "data": research_tools.risk_status(), "execution_enabled": False}


def model_config_payload(config: RuntimeConfig) -> dict:
    return {
        "enabled": config.ai_model_enabled,
        "provider": config.ai_provider,
        "base_url": config.ai_base_url,
        "model": config.ai_model,
        "api_key_env": config.ai_api_key_env,
        "temperature": config.ai_temperature,
        "max_tokens": config.ai_max_tokens,
        "execution_enabled": False,
        "api_key": "[provided via environment]" if config.ai_api_key_env else "[not configured]",
        "api_key_configured": bool(config.ai_api_key_env and os.getenv(config.ai_api_key_env)),
    }


def schedule_service_restart() -> dict:
    """Restart this local paper-only service with its original CLI arguments."""
    # sys.argv[0] may point to __main__.py when started with `python -m`.
    # Re-invoke the package module so relative imports remain valid.
    command = [sys.executable, "-m", "quant_platform", *sys.argv[1:]]
    managed_by_launchd = os.getenv("QUANT_PLATFORM_LAUNCHD") == "1"

    def restart() -> None:
        time.sleep(0.4)
        if not managed_by_launchd:
            subprocess.Popen(command, start_new_session=True, close_fds=True)
        os._exit(0)

    threading.Thread(target=restart, daemon=True).start()
    return {
        "message": "服务正在重启，请等待 2-5 秒后刷新页面",
        "execution_enabled": False,
        "supervisor": "launchd" if managed_by_launchd else "self-restart",
        "command": [sys.executable, "-m", "quant_platform", "serve"],
    }


def api_payload(
    path: str,
    config: RuntimeConfig,
    control_plane=None,
    research_tools=None,
    simulation_registry: PaperSimulationRegistry | None = None,
):
    parsed = urlsplit(path)
    path = parsed.path
    query = parse_qs(parsed.query)
    if path == "/health":
        return health_snapshot(
            config.environment,
            config.live_trading_enabled,
            config.ai_execution_tools_enabled,
        )
    if path == "/ai-config":
        return model_config_payload(config)
    if path == "/settings/tushare-token":
        return tushare_token_status()
    if path == "/orders" and control_plane is not None:
        return {"orders": control_plane.orders(), "execution_enabled": False}
    if path == "/audit-events" and control_plane is not None:
        order_id = query.get("order_id", [None])[0]
        return {"events": control_plane.audit_events(order_id), "execution_enabled": False}
    if path == "/risk" and control_plane is not None:
        return {**control_plane.risk_state(), "execution_enabled": False}
    if path == "/system" and control_plane is not None:
        return control_plane.system_state()
    if path == "/adapters" and control_plane is not None:
        return {"adapters": control_plane.adapter_health(), "execution_enabled": False}
    if path == "/experiments" and control_plane is not None:
        return {"experiments": control_plane.experiments(), "execution_enabled": False}
    if path == "/strategies" and control_plane is not None:
        return {"strategies": control_plane.strategies(), "execution_enabled": False}
    if path == "/portfolio" and control_plane is not None:
        return {**control_plane.portfolio_state(), "execution_enabled": False}
    if path == "/research" and research_tools is not None:
        return {"tools": research_tools.catalog(), "execution_enabled": False}
    if path == "/market-data" and research_tools is not None:
        symbol = query.get("symbol", [None])[0]
        raw_limit = query.get("limit", [None])[0]
        limit = None if raw_limit is None else int(raw_limit)
        return {"data": research_tools.market_data(symbol, limit), "execution_enabled": False}
    if path == "/research/portfolio" and research_tools is not None:
        return {"portfolio": research_tools.portfolio_state(), "execution_enabled": False}
    if path == "/run-history":
        return {"runs": get_store().all(), "execution_enabled": False, "paper_only": True}
    if path == "/simulation/state" and simulation_registry is not None:
        account_id = query.get("account_id", [None])[0]
        if account_id is None:
            return {
                "accounts": simulation_registry.list_states(),
                "paper_only": True,
                "execution_enabled": False,
            }
        return simulation_registry.get(account_id).state()
    return None


_JSON_POST_ROUTES = {
    "/settings/tushare-token", "/agent/research", "/order-features",
    "/simulation-risk", "/factor-analysis", "/factor-panel-analysis",
    "/market-state", "/live-readiness", "/simulation/create",
    "/simulation/order", "/simulation/market", "/simulation/control",
    "/simulation/reconciliation", "/simulation/acceptance", "/backtest",
    "/simulation/portfolio-report", "/simulation/delete", "/simulation/account-report",
    "/phase9/run", "/factor/symbol", "/run-history/delete",
    "/validation/forward",
}


def validate_local_post_request(route: str, headers, csrf_token: str) -> None:
    """Reject browser cross-origin mutation while preserving trusted local CLI use."""
    host = str(headers.get("Host", "")).strip()
    try:
        hostname = urlsplit(f"//{host}").hostname or ""
        ensure_loopback_host(hostname)
    except ValueError as error:
        raise PermissionError("POST Host must resolve to loopback") from error

    if route in _JSON_POST_ROUTES:
        content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("POST Content-Type must be application/json")

    origin = str(headers.get("Origin", "")).strip()
    referer = str(headers.get("Referer", "")).strip()
    if origin:
        _validate_browser_source(origin, host, "origin")
    elif referer:
        _validate_browser_source(referer, host, "referer")

    if origin or referer:
        supplied_token = str(headers.get("X-CSRF-Token", ""))
        if not supplied_token or not hmac.compare_digest(supplied_token, csrf_token):
            raise PermissionError("browser POST requires a valid CSRF token")


def _validate_browser_source(source: str, host: str, label: str) -> None:
    """Reject a browser Origin/Referer that is not the local loopback console."""
    parsed = urlsplit(source)
    if parsed.scheme != "http" or parsed.netloc.lower() != host.lower():
        raise PermissionError(f"browser POST {label} must match the local console")
    try:
        ensure_loopback_host(parsed.hostname or "")
    except ValueError as error:
        raise PermissionError(f"browser POST {label} must be loopback") from error


def make_handler(
    config: RuntimeConfig,
    control_plane=None,
    research_tools=None,
    simulation_registry: PaperSimulationRegistry | None = None,
):
    simulation_registry = simulation_registry or PaperSimulationRegistry()
    forward_validation_tracker = ForwardValidationTracker()
    csrf_token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path == "/csrf-token":
                payload = json.dumps({"csrf_token": csrf_token, "execution_enabled": False}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if urlsplit(self.path).path == "/":
                page = Path(__file__).resolve().parents[2] / "web" / "index.html"
                if page.exists():
                    payload = page.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
            body = api_payload(
                self.path,
                config,
                control_plane,
                research_tools,
                simulation_registry,
            )
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            route = urlsplit(self.path).path
            try:
                validate_local_post_request(route, self.headers, csrf_token)
            except (PermissionError, ValueError) as error:
                payload = json.dumps({"error": str(error), "execution_enabled": False}).encode("utf-8")
                self.send_response(403 if isinstance(error, PermissionError) else 415)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/settings/tushare-token":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 2048:
                        raise ValueError("invalid request size")
                    request = json.loads(self.rfile.read(length))
                    body = save_tushare_token(request.get("token", ""))
                    status = 200
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/service/restart":
                payload = json.dumps(schedule_service_restart(), ensure_ascii=False).encode("utf-8")
                self.send_response(202)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/paper-demo":
                payload = json.dumps(paper_demo_payload()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/agent/research":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length))
                    prompt = request.get("prompt", "")
                    if config.ai_model_enabled:
                        body = model_research_payload(prompt, config, research_tools)
                    else:
                        body = research_agent_payload(prompt, research_tools)
                    status = 200
                except Exception as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            analysis_routes = {
                "/order-features": order_features_payload,
                "/simulation-risk": simulation_risk_payload,
                "/factor-analysis": factor_analysis_payload,
                "/factor-panel-analysis": factor_panel_analysis_payload,
                "/market-state": market_state_payload,
                "/live-readiness": live_readiness_payload,
            }
            if route in analysis_routes:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = analysis_routes[route](json.loads(self.rfile.read(length)))
                    status = 200
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            simulation_routes = {
                "/simulation/create": simulation_create_payload,
                "/simulation/order": simulation_order_payload,
                "/simulation/market": simulation_market_payload,
                "/simulation/control": simulation_control_payload,
                "/simulation/reconciliation": simulation_reconciliation_payload,
                "/simulation/acceptance": simulation_acceptance_payload,
                "/simulation/delete": simulation_delete_payload,
                "/simulation/account-report": simulation_account_report_payload,
            }
            if route in simulation_routes:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = simulation_routes[route](
                        json.loads(self.rfile.read(length)),
                        simulation_registry,
                    )
                    status = 200
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/simulation/portfolio-report":
                try:
                    body = forward_validation_tracker.report()
                    status = 200
                except (ValueError, TypeError, RuntimeError) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/phase9/run":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = phase9_payload(
                        json.loads(self.rfile.read(length)),
                        control_plane,
                        simulation_registry,
                        config,
                    )
                    status = 200
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/run-history/delete":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = run_history_delete_payload(
                        json.loads(self.rfile.read(length)), simulation_registry
                    )
                    status = 200
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/validation/forward":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = forward_validation_payload(
                        json.loads(self.rfile.read(length)), forward_validation_tracker
                    )
                    status = 200
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/factor/symbol":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = factor_symbol_payload(json.loads(self.rfile.read(length)), config)
                    status = 200
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as error:
                    body = {"error": str(error), "execution_enabled": False}
                    status = 400
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route != "/backtest":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = backtest_payload(json.loads(self.rfile.read(length)))
                status = 200
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                body = {"error": str(error), "execution_enabled": False}
                status = 400
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    return Handler


def ensure_loopback_host(host: str) -> str:
    """Reject accidental LAN exposure until authenticated remote access exists."""
    normalized = host.strip().lower()
    if normalized == "localhost":
        return host
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as error:
        raise ValueError("HTTP service host must be a loopback address") from error
    if not address.is_loopback:
        raise ValueError("HTTP service host must be a loopback address")
    return host


def serve(
    config: RuntimeConfig,
    host: str = "127.0.0.1",
    port: int = 8080,
    control_plane=None,
    research_tools=None,
    simulation_registry: PaperSimulationRegistry | None = None,
) -> None:
    host = ensure_loopback_host(host)
    ThreadingHTTPServer(
        (host, port),
        make_handler(config, control_plane, research_tools, simulation_registry),
    ).serve_forever()
