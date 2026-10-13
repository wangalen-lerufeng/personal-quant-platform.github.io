import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from quant_platform.config import RuntimeConfig
from quant_platform.control import ControlPlane
from quant_platform.domain import OrderIntent, OrderSide
from quant_platform.http_api import (
    api_payload,
    backtest_payload,
    execute_research_tool_call,
    extract_research_tool_call,
    extract_research_tool_calls,
    factor_analysis_payload,
    factor_panel_analysis_payload,
    market_state_payload,
    model_research_payload,
    plan_stock_research_calls,
    validate_local_post_request,
    live_readiness_payload,
    order_features_payload,
    paper_demo_payload,
    research_agent_payload,
    simulation_create_payload,
    simulation_acceptance_payload,
    simulation_reconciliation_payload,
    simulation_control_payload,
    simulation_market_payload,
    simulation_order_payload,
    simulation_risk_payload,
    ensure_loopback_host,
    build_fundamental_analysis_result,
)
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.research_tools import ResearchTools
from quant_platform.domain import AssetClass, MarketEvent
from quant_platform.market_data import InMemoryMarketDataFeed
from datetime import date, datetime, timezone
from quant_platform.storage import SimulationAccountStore, SqliteAuditStore
from quant_platform.simulation import PaperSimulationRegistry


class HttpApiTests(unittest.TestCase):
    def setUp(self):
        store = SqliteAuditStore(":memory:")
        control = ControlPlane(RuntimeConfig(), RiskEngine(RiskLimits()), store)
        control.record_order("order-1", OrderIntent("EURUSD", OrderSide.BUY, 0.01, "demo"), "accepted")
        self.config = RuntimeConfig()
        self.control = control

    def test_health_and_orders_routes_return_json_payloads(self):
        health = api_payload("/health", self.config, self.control)
        orders = api_payload("/orders", self.config, self.control)

        self.assertEqual(health["status"], "ok")
        self.assertEqual(orders["orders"][0]["symbol"], "EURUSD")
        self.assertFalse(orders["execution_enabled"])
        self.assertFalse(api_payload("/risk", self.config, self.control)["execution_enabled"])
        self.assertFalse(api_payload("/adapters", self.config, self.control)["execution_enabled"])

    def test_ai_config_is_read_only_and_does_not_expose_secret(self):
        payload = api_payload("/ai-config", self.config, self.control)
        self.assertFalse(payload["enabled"])
        self.assertFalse(payload["execution_enabled"])
        self.assertNotIn("sk-", payload["api_key"])

    def test_system_route_exposes_execution_safety_state(self):
        payload = api_payload("/system", self.config, self.control)

        self.assertFalse(payload["execution_enabled"])
        self.assertTrue(payload["paper_only"])

    def test_audit_events_route_supports_order_filter(self):
        payload = api_payload("/audit-events?order_id=order-1", self.config, self.control)

        self.assertEqual(payload["events"][0]["order_id"], "order-1")
        self.assertFalse(payload["execution_enabled"])

    def test_unknown_route_returns_none(self):
        self.assertIsNone(api_payload("/unknown", self.config, self.control))

    def test_research_backtest_payload_returns_metrics_without_execution(self):
        result = backtest_payload({
            "events": [
                {"symbol": "EURUSD", "asset_class": "fx", "price": 1.08, "timestamp": "2026-01-01T00:00:00+00:00"},
                {"symbol": "EURUSD", "asset_class": "fx", "price": 1.09, "timestamp": "2026-01-02T00:00:00+00:00"},
                {"symbol": "EURUSD", "asset_class": "fx", "price": 1.10, "timestamp": "2026-01-03T00:00:00+00:00"},
            ],
            "lookback": 1,
            "quantity": 1,
        })

        self.assertEqual(result["trade_count"], 1)
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["research_only"])

    def test_research_backtest_rejects_empty_events(self):
        with self.assertRaises(ValueError):
            backtest_payload({"events": []})

    def test_order_features_payload_is_read_only(self):
        result = order_features_payload({"trades": [{"side": "buy", "price": 10, "quantity": 2}]})
        self.assertEqual(result["trade_count"], 1)
        self.assertTrue(result["research_only"])
        self.assertFalse(result["execution_enabled"])

    def test_simulation_risk_payload_reports_threshold_breaches_without_execution(self):
        result = simulation_risk_payload({
            "initial_equity": 10_000,
            "equity_series": [10_500, 9_400],
            "max_daily_loss": 500,
            "max_drawdown": 1_000,
        })

        self.assertEqual(result["latest"]["risk_level"], "halt")
        self.assertIn("daily_loss_limit", result["latest"]["alerts"])
        self.assertIn("drawdown_limit", result["latest"]["alerts"])
        self.assertEqual(len(result["history"]), 2)
        self.assertTrue(result["research_only"])
        self.assertFalse(result["execution_enabled"])

    def test_factor_analysis_payload_calculates_ic_without_execution(self):
        result = factor_analysis_payload({
            "records": [
                {"factor": 1, "forward_return": 0.01},
                {"factor": 2, "forward_return": 0.02},
                {"factor": 3, "forward_return": 0.03},
                {"factor": 4, "forward_return": 0.04},
                {"factor": 5, "forward_return": 0.05},
                {"factor": 6, "forward_return": 0.06},
            ],
            "groups": 3,
        })

        self.assertAlmostEqual(result["ic"], 1.0)
        self.assertAlmostEqual(result["rank_ic"], 1.0)
        self.assertEqual(result["observations"], 6)
        self.assertTrue(result["research_only"])
        self.assertFalse(result["execution_enabled"])

    def test_factor_panel_payload_reports_time_series_without_execution(self):
        records = []
        for day in ("2026-01-02", "2026-01-05"):
            for index, symbol in enumerate(("A", "B", "C", "D"), start=1):
                records.append({
                    "timestamp": day,
                    "symbol": symbol,
                    "factor": index,
                    "forward_return": index / 100,
                })

        result = factor_panel_analysis_payload({
            "records": records,
            "return_columns": ["forward_return"],
            "groups": 2,
        })

        self.assertEqual(result["date_count"], 2)
        self.assertIn("forward_return", result["ic_series"])
        self.assertTrue(result["research_only"])
        self.assertFalse(result["execution_enabled"])

        tool_result = execute_research_tool_call({
            "tool": "analyze_factor_panel",
            "arguments": {"records": records, "return_columns": ["forward_return"], "groups": 2},
        })
        self.assertEqual(tool_result["tool"], "analyze_factor_panel")
        self.assertFalse(tool_result["execution_enabled"])

    def test_backtest_payload_can_enforce_exchange_constraints(self):
        result = backtest_payload({
            "events": [
                {
                    "timestamp": "2026-01-05T08:00:00+08:00",
                    "symbol": "510300",
                    "asset_class": "etf",
                    "price": 4.0,
                },
                {
                    "timestamp": "2026-01-05T09:31:00+08:00",
                    "symbol": "510300",
                    "asset_class": "etf",
                    "price": 4.1,
                },
            ],
            "lookback": 1,
            "quantity": 100,
            "enforce_market_rules": True,
        })

        self.assertEqual(result["restriction_count"], 1)
        self.assertEqual(result["restrictions"][0]["reason"], "market_closed")
        self.assertFalse(result["execution_enabled"])

    def test_backtest_payload_can_enable_audited_futures_rollover(self):
        result = backtest_payload({
            "events": [
                {"timestamp": "2026-01-14T09:31:00+08:00", "symbol": "IF2601", "asset_class": "futures", "price": 4000},
                {"timestamp": "2026-01-14T10:00:00+08:00", "symbol": "IF2601", "asset_class": "futures", "price": 4010},
                {"timestamp": "2026-01-15T09:31:00+08:00", "symbol": "IF2602", "asset_class": "futures", "price": 4020},
            ],
            "lookback": 1,
            "quantity": 2,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "enforce_market_rules": True,
            "auto_rollover": True,
            "rollover_rules": [{
                "expiring_symbol": "IF2601",
                "successor_symbol": "IF2602",
                "effective_at": "2026-01-15T09:00:00+08:00",
            }],
        })

        self.assertEqual(result["rollover_count"], 1)
        self.assertEqual(result["rollovers"][0]["to_symbol"], "IF2602")
        self.assertFalse(result["execution_enabled"])

    def test_market_state_payload_classifies_bars_without_execution(self):
        result = market_state_payload({
            "bars": [
                {"close": 10.0},
                {"close": 10.5},
                {"close": 11.0},
                {"close": 11.5},
                {"close": 12.0},
            ],
            "window": 3,
        })

        self.assertEqual(result["state"], "trend")
        self.assertEqual(result["evidence"]["window"], 3)
        self.assertTrue(result["research_only"])
        self.assertFalse(result["execution_enabled"])

    def test_simulation_api_payloads_create_order_match_and_query_state(self):
        registry = PaperSimulationRegistry()
        created = simulation_create_payload({
            "account_id": "paper-api",
            "initial_cash": 10_000,
            "max_drawdown": 1_000,
            "max_consecutive_losses": 3,
            "max_participation_rate": 1.0,
        }, registry)
        submitted = simulation_order_payload({
            "account_id": "paper-api",
            "symbol": "510300.SH",
            "side": "buy",
            "quantity": 100,
            "strategy_id": "cta",
            "asset_class": "etf",
            "reference_price": 4.0,
        }, registry)
        matched = simulation_market_payload({
            "account_id": "paper-api",
            "symbol": "510300.SH",
            "price": 4.0,
            "volume": 100,
        }, registry)
        state = api_payload(
            "/simulation/state?account_id=paper-api",
            self.config,
            self.control,
            None,
            registry,
        )

        self.assertEqual(created["account"]["cash"], 10_000)
        self.assertEqual(submitted["status"], "accepted")
        self.assertEqual(matched["fills"][0]["quantity"], 100)
        self.assertEqual(state["orders"][submitted["order_id"]], "filled")
        self.assertTrue(state["paper_only"])
        self.assertFalse(state["execution_enabled"])

    def test_simulation_control_records_losses_and_manual_kill_switch(self):
        registry = PaperSimulationRegistry()
        simulation_create_payload({
            "account_id": "paper-control",
            "initial_cash": 10_000,
            "max_consecutive_losses": 2,
        }, registry)

        simulation_control_payload({
            "account_id": "paper-control",
            "action": "record_trade_pnl",
            "pnl": -10,
        }, registry)
        loss_state = simulation_control_payload({
            "account_id": "paper-control",
            "action": "record_trade_pnl",
            "pnl": -5,
        }, registry)
        kill_state = simulation_control_payload({
            "account_id": "paper-control",
            "action": "kill_switch",
            "reason": "operator_requested",
        }, registry)

        self.assertIn("consecutive_loss_limit", loss_state["operational_risk"]["alerts"])
        self.assertIn("operator_requested", kill_state["operational_risk"]["alerts"])
        self.assertFalse(kill_state["execution_enabled"])

    def test_simulation_reconciliation_and_acceptance_are_read_only(self):
        with TemporaryDirectory() as directory:
            store = SimulationAccountStore(Path(directory) / "simulation.db")
            registry = PaperSimulationRegistry(store)
            simulation_create_payload({
                "account_id": "paper-evidence",
                "initial_cash": 10_000,
                "max_participation_rate": 1.0,
            }, registry)
            simulation_order_payload({
                "account_id": "paper-evidence",
                "symbol": "510300.SH",
                "side": "buy",
                "quantity": 100,
                "strategy_id": "cta",
                "asset_class": "etf",
                "reference_price": 4.0,
            }, registry)
            simulation_market_payload({
                "account_id": "paper-evidence",
                "symbol": "510300.SH",
                "price": 4.0,
                "volume": 100,
            }, registry)

            reconciliation = simulation_reconciliation_payload(
                {"account_id": "paper-evidence"}, registry
            )
            acceptance = simulation_acceptance_payload(
                {"account_id": "paper-evidence", "minimum_trading_days": 30}, registry
            )

            self.assertTrue(reconciliation["passed"])
            self.assertEqual(acceptance["paper_trading_days"], 1)
            self.assertIn("minimum_paper_trading_days_not_met", acceptance["blockers"])
            self.assertFalse(reconciliation["execution_enabled"])
            self.assertFalse(acceptance["execution_enabled"])

    def test_live_readiness_payload_never_enables_execution(self):
        result = live_readiness_payload({
            "out_of_sample_passed": True,
            "paper_trading_days": 45,
            "reconciliation_passed": True,
            "risk_limits_configured": True,
            "manual_approval_id": "approval-001",
        })

        self.assertTrue(result["eligible_to_request"])
        self.assertFalse(result["live_trading_enabled"])
        self.assertFalse(result["execution_enabled"])

    def test_paper_demo_runs_without_execution(self):
        result = paper_demo_payload()
        self.assertEqual(result["paper_results"], 1)
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["paper_only"])

    def test_research_agent_orchestrates_read_only_tools(self):
        result = research_agent_payload("请回测 EURUSD 动量策略")
        self.assertIn("run_backtest", " ".join(result["orchestration"]))
        self.assertTrue(result["tools_are_read_only"])
        self.assertFalse(result["execution_enabled"])

    def test_research_agent_routes_factor_questions_to_factor_tool(self):
        result = research_agent_payload("分析这个因子的 IC 和分组收益")

        self.assertIn("analyze_factor", " ".join(result["orchestration"]))
        self.assertFalse(result["execution_enabled"])

    def test_research_agent_routes_advanced_factor_questions_to_panel_tool(self):
        result = research_agent_payload("分析因子的 ICIR、衰减、容量和样本外表现")

        self.assertIn("analyze_factor_panel", " ".join(result["orchestration"]))
        self.assertFalse(result["execution_enabled"])

    def test_research_tool_call_is_parsed_and_unknown_tools_are_rejected(self):
        call = extract_research_tool_call('```json {"tool":"get_market_data","arguments":{"symbol":"EURUSD"}} ```')
        self.assertEqual(call["tool"], "get_market_data")
        result = execute_research_tool_call({"tool": "submit_order", "arguments": {}}, None)
        self.assertIn("not allowed", result["error"])

    def test_multiple_research_tool_calls_are_parsed_in_order(self):
        answer = """
        {"tool":"get_market_data","arguments":{"dataset":"stock_basic"}}
        {"tool":"get_market_data","arguments":{"dataset":"income"}}
        """
        calls = extract_research_tool_calls(answer)
        self.assertEqual([call["arguments"]["dataset"] for call in calls], ["stock_basic", "income"])

    def test_fundamental_tool_alias_is_normalized_to_market_data(self):
        calls = extract_research_tool_calls(
            '{"tool":"get_fundamental_data","arguments":{"provider":"tushare","dataset":"income"}}'
        )
        self.assertEqual(calls[0]["tool"], "get_market_data")

    def test_deepseek_dsml_tool_calls_are_parsed(self):
        answer = '''
        <｜｜DSML｜｜tool_calls>
        <｜｜DSML｜｜invoke name="get_market_data">
        <｜｜DSML｜｜parameter name="provider" string="true">tushare</｜｜DSML｜｜parameter>
        <｜｜DSML｜｜parameter name="symbol" string="true">603893.SH</｜｜DSML｜｜parameter>
        <｜｜DSML｜｜parameter name="dataset" string="true">income</｜｜DSML｜｜parameter>
        </｜｜DSML｜｜invoke>
        </｜｜DSML｜｜tool_calls>
        '''
        calls = extract_research_tool_calls(answer)

        self.assertEqual(calls, [{
            "tool": "get_market_data",
            "arguments": {"provider": "tushare", "symbol": "603893.SH", "dataset": "income"},
        }])

    def test_stock_deep_research_has_deterministic_dataset_plan(self):
        calls = plan_stock_research_calls(
            "请深度研究603893.SH，研究截止日为2026-10-13，检查stock_basic、daily、daily_basic、income、balancesheet、cashflow、fina_indicator。"
        )

        self.assertEqual(
            [call["arguments"]["dataset"] for call in calls],
            ["stock_basic", "daily", "daily_basic", "income", "balancesheet", "cashflow", "fina_indicator"],
        )
        self.assertTrue(all(call["tool"] == "get_market_data" for call in calls))
        self.assertTrue(all(call["arguments"]["end_date"] == "2026-10-13" for call in calls))

    def test_deep_research_executes_tool_call_generated_after_third_round(self):
        tool_json = '{"tool":"portfolio_state","arguments":{}}'
        client = unittest.mock.Mock()
        client.complete.return_value = tool_json
        client.complete_with_tool_result.side_effect = [tool_json, tool_json, tool_json, "最终研究报告"]
        with patch("quant_platform.http_api.OpenAICompatibleChatClient", return_value=client), patch(
            "quant_platform.http_api.execute_research_tool_call_safely",
            return_value={"tool": "portfolio_state", "data": {"equity": 1}},
        ) as execute:
            result = model_research_payload("深度研究", RuntimeConfig())

        self.assertEqual(execute.call_count, 4)
        self.assertEqual(result["answer"], "最终研究报告")

    def test_financial_tool_results_receive_programmatic_analysis(self):
        result = build_fundamental_analysis_result([
            {"dataset": "income", "data": [{"end_date": "20241231", "ann_date": "20250320", "revenue": 100, "n_income_attr_p": 10}]},
            {"dataset": "cashflow", "data": [{"end_date": "20241231", "ann_date": "20250320", "n_cashflow_act": 8}]},
        ], date(2025, 12, 31))

        self.assertEqual(result["dataset"], "fundamental_analysis")
        self.assertAlmostEqual(result["data"]["computed_metrics"]["cash_flow_quality"][0]["value"], 0.8)
        self.assertFalse(result["execution_enabled"])

    def test_tushare_non_price_dataset_uses_dataset_provider_not_daily(self):
        with patch("quant_platform.http_api.TushareDatasetProvider") as provider:
            provider.return_value.fetch_records.return_value = [{"ts_code": "003019.SZ", "name": "sample"}]
            result = execute_research_tool_call({
                "tool": "get_market_data",
                "arguments": {
                    "provider": "tushare",
                    "dataset": "stock_basic",
                    "asset_class": "equity",
                    "symbol": "003019.SZ",
                    "start_date": "2019-01-01",
                    "end_date": "2025-12-31",
                },
            }, object())

        provider.return_value.fetch_records.assert_called_once_with(
            "stock_basic", "003019.SZ", "2019-01-01", "2025-12-31"
        )
        self.assertEqual(result["dataset"], "stock_basic")
        self.assertEqual(result["data"][0]["ts_code"], "003019.SZ")

    def test_model_tool_cannot_select_a_credential_environment_variable(self):
        with patch("quant_platform.http_api.TushareProvider") as provider:
            result = execute_research_tool_call({
                "tool": "get_market_data",
                "arguments": {
                    "provider": "tushare",
                    "symbol": "510300.SH",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-31",
                    "token_env": "AI_API_KEY",
                },
            }, object())

        self.assertIn("credential selector", result["error"])
        provider.assert_not_called()

    def test_tushare_equity_tool_uses_free_stock_daily_provider(self):
        event = MarketEvent(
            "000001.SZ",
            AssetClass.EQUITY,
            10.5,
            datetime(2026, 1, 5, tzinfo=timezone.utc),
        )
        with patch("quant_platform.http_api.TushareStockDailyProvider") as provider:
            provider.return_value.fetch.return_value = [event]
            result = execute_research_tool_call({
                "tool": "get_market_data",
                "arguments": {
                    "provider": "tushare",
                    "dataset": "daily",
                    "asset_class": "equity",
                    "symbol": "000001.SZ",
                    "start_date": "2025-01-01",
                    "end_date": "2026-01-05",
                },
            }, object())

        provider.assert_called_once_with("TUSHARE_TOKEN")
        self.assertEqual(result["data"][0]["asset_class"], "equity")
        self.assertEqual(result["dataset"], "daily")

    def test_tushare_equity_backtest_constructs_momentum_strategy(self):
        events = [
            MarketEvent(
                "000001.SZ",
                AssetClass.EQUITY,
                price,
                datetime(2026, 1, day, tzinfo=timezone.utc),
            )
            for day, price in ((2, 10.0), (3, 10.2), (4, 10.4), (5, 10.1))
        ]
        with patch("quant_platform.http_api.TushareStockDailyProvider") as provider:
            provider.return_value.fetch.return_value = events
            result = execute_research_tool_call({
                "tool": "run_backtest",
                "arguments": {
                    "provider": "tushare",
                    "dataset": "daily",
                    "asset_class": "equity",
                    "symbol": "000001.SZ",
                    "start_date": "2026-01-02",
                    "end_date": "2026-01-05",
                    "lookback": 2,
                    "quantity": 100,
                },
            })

        self.assertNotIn("error", result)
        self.assertTrue(result["research_only"])

    def test_http_service_rejects_non_loopback_bind_addresses(self):
        self.assertEqual(ensure_loopback_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(ensure_loopback_host("localhost"), "localhost")
        with self.assertRaisesRegex(ValueError, "loopback"):
            ensure_loopback_host("0.0.0.0")

    def test_browser_post_requires_same_origin_and_csrf_token(self):
        headers = {
            "Host": "127.0.0.1:8080",
            "Origin": "http://127.0.0.1:8080",
            "Content-Type": "application/json",
            "X-CSRF-Token": "token-123",
        }
        validate_local_post_request("/agent/research", headers, "token-123")

        with self.assertRaisesRegex(PermissionError, "CSRF"):
            validate_local_post_request("/agent/research", {**headers, "X-CSRF-Token": "wrong"}, "token-123")
        with self.assertRaisesRegex(PermissionError, "origin"):
            validate_local_post_request("/agent/research", {**headers, "Origin": "https://evil.example"}, "token-123")

    def test_json_post_rejects_plain_text_and_non_loopback_host(self):
        with self.assertRaisesRegex(ValueError, "application/json"):
            validate_local_post_request(
                "/settings/tushare-token",
                {"Host": "127.0.0.1:8080", "Content-Type": "text/plain"},
                "token-123",
            )
        with self.assertRaisesRegex(PermissionError, "loopback"):
            validate_local_post_request(
                "/agent/research",
                {"Host": "attacker.example", "Content-Type": "application/json"},
                "token-123",
            )

    def test_local_command_line_json_post_remains_supported_without_origin(self):
        validate_local_post_request(
            "/agent/research",
            {"Host": "127.0.0.1:8080", "Content-Type": "application/json"},
            "token-123",
        )

    def test_browser_post_with_referer_only_requires_csrf_token(self):
        headers = {
            "Host": "127.0.0.1:8080",
            "Referer": "http://127.0.0.1:8080/",
            "Content-Type": "application/json",
        }
        with self.assertRaisesRegex(PermissionError, "CSRF"):
            validate_local_post_request("/agent/research", headers, "token-123")
        with self.assertRaisesRegex(PermissionError, "CSRF"):
            validate_local_post_request(
                "/agent/research", {**headers, "X-CSRF-Token": "wrong"}, "token-123"
            )
        validate_local_post_request(
            "/agent/research", {**headers, "X-CSRF-Token": "token-123"}, "token-123"
        )

    def test_browser_post_rejects_cross_origin_referer(self):
        with self.assertRaisesRegex(PermissionError, "referer"):
            validate_local_post_request(
                "/agent/research",
                {
                    "Host": "127.0.0.1:8080",
                    "Referer": "http://evil.example/page",
                    "Content-Type": "application/json",
                    "X-CSRF-Token": "token-123",
                },
                "token-123",
            )

    def test_browser_post_rejects_non_http_referer_scheme(self):
        with self.assertRaisesRegex(PermissionError, "referer"):
            validate_local_post_request(
                "/agent/research",
                {
                    "Host": "127.0.0.1:8080",
                    "Referer": "https://127.0.0.1:8080/",
                    "Content-Type": "application/json",
                    "X-CSRF-Token": "token-123",
                },
                "token-123",
            )

    def test_backtest_tool_requires_historical_data_instead_of_fabricating_results(self):
        result = execute_research_tool_call({"tool": "run_backtest", "arguments": {}}, None)
        self.assertIn("no historical market data", result["error"])

    def test_ai_whitelist_can_execute_local_read_only_analytics(self):
        factor_result = execute_research_tool_call({
            "tool": "analyze_factor",
            "arguments": {
                "records": [
                    {"factor": 1, "forward_return": 0.01},
                    {"factor": 2, "forward_return": 0.02},
                    {"factor": 3, "forward_return": 0.03},
                    {"factor": 4, "forward_return": 0.04},
                ],
                "groups": 2,
            },
        })
        risk_result = execute_research_tool_call({
            "tool": "monitor_simulation_risk",
            "arguments": {
                "initial_equity": 10_000,
                "equity_series": [9_000],
                "max_daily_loss": 500,
            },
        })

        self.assertEqual(factor_result["tool"], "analyze_factor")
        self.assertAlmostEqual(factor_result["data"]["rank_ic"], 1.0)
        self.assertEqual(risk_result["data"]["latest"]["risk_level"], "halt")
        self.assertFalse(factor_result["execution_enabled"])
        self.assertFalse(risk_result["execution_enabled"])

    def test_query_string_does_not_break_route_lookup(self):
        self.assertEqual(
            api_payload("/health?format=json", self.config, self.control)["status"],
            "ok",
        )

    def test_research_route_exposes_read_only_tool_catalog(self):
        tools = ResearchTools(self.control)

        payload = api_payload("/research", self.config, self.control, tools)

        self.assertIn("system_status", payload["tools"])
        self.assertNotIn("submit_order", payload["tools"])

    def test_market_data_route_is_read_only_and_supports_symbol_filter(self):
        feed = InMemoryMarketDataFeed(
            [MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc))]
        )
        tools = ResearchTools(self.control, market_data_feed=feed)

        payload = api_payload("/market-data?symbol=EURUSD", self.config, self.control, tools)

        self.assertEqual(payload["data"][0]["symbol"], "EURUSD")
        self.assertFalse(payload["execution_enabled"])

    def test_market_data_route_supports_limit(self):
        feed = InMemoryMarketDataFeed([
            MarketEvent("EURUSD", AssetClass.FX, 1.08, datetime(2026, 1, 1, tzinfo=timezone.utc)),
            MarketEvent("EURUSD", AssetClass.FX, 1.09, datetime(2026, 1, 2, tzinfo=timezone.utc)),
        ])
        tools = ResearchTools(self.control, market_data_feed=feed)

        payload = api_payload("/market-data?symbol=EURUSD&limit=1", self.config, self.control, tools)

        self.assertEqual(len(payload["data"]), 1)
        self.assertEqual(payload["data"][0]["price"], 1.09)

    def test_portfolio_route_is_read_only(self):
        payload = api_payload("/portfolio", self.config, self.control)

        self.assertEqual(payload["positions"], {})
        self.assertEqual(payload["equity"], 0.0)
        self.assertFalse(payload["execution_enabled"])

    def test_research_portfolio_route_is_read_only(self):
        tools = ResearchTools(self.control)

        payload = api_payload("/research/portfolio", self.config, self.control, tools)

        self.assertEqual(payload["portfolio"]["positions"], {})
        self.assertFalse(payload["execution_enabled"])

    def test_strategies_route_is_read_only(self):
        from quant_platform.strategy_registry import StrategyRegistry

        self.control.strategy_registry = StrategyRegistry([("momentum", {"execution_enabled": False})])

        payload = api_payload("/strategies", self.config, self.control)

        self.assertEqual(payload["strategies"][0]["strategy_id"], "momentum")
        self.assertFalse(payload["strategies"][0]["execution_enabled"])
        self.assertFalse(payload["execution_enabled"])
