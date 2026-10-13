import unittest
from pathlib import Path


class WebUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = (
            Path(__file__).resolve().parents[1] / "web" / "index.html"
        ).read_text(encoding="utf-8")

    def test_research_console_exposes_risk_factor_and_market_state_tools(self):
        for element_id in (
            "runRiskMonitor",
            "riskMonitorOutput",
            "runFactorAnalysis",
            "factorAnalysisOutput",
            "runFactorPanelAnalysis",
            "factorPanelOutput",
            "runMarketState",
            "marketStateOutput",
        ):
            self.assertIn(f'id="{element_id}"', self.page)

        for route in ("/simulation-risk", "/factor-analysis", "/factor-panel-analysis", "/market-state"):
            self.assertIn(route, self.page)

    def test_backtest_console_exposes_costs_and_exchange_rules(self):
        for element_id in ("commissionRate", "slippageRate", "enforceMarketRules"):
            self.assertIn(f'id="{element_id}"', self.page)
        self.assertIn("restriction_count", self.page)

    def test_research_panels_keep_execution_disabled_language_visible(self):
        self.assertIn("只做研究，不会触发下单", self.page)
        self.assertIn("本页面不提供下单能力", self.page)

    def test_paper_simulation_console_exposes_account_order_market_and_stop_controls(self):
        for element_id in (
            "createSimulation",
            "submitSimulationOrder",
            "pushSimulationMarket",
            "stopSimulation",
            "simulationOutput",
            "runSimulationReconciliation",
            "runSimulationAcceptance",
            "simulationAcceptanceOutput",
        ):
            self.assertIn(f'id="{element_id}"', self.page)
        for route in (
            "/simulation/create",
            "/simulation/order",
            "/simulation/market",
            "/simulation/control",
            "/simulation/reconciliation",
            "/simulation/acceptance",
        ):
            self.assertIn(route, self.page)
        self.assertIn("人工停止", self.page)

    def test_console_has_hash_routed_page_navigation(self):
        self.assertIn('id="pageNav"', self.page)
        for page in (
            "dashboard",
            "data",
            "research",
            "backtest",
            "simulation",
            "risk",
            "settings",
        ):
            self.assertIn(f'data-page-target="{page}"', self.page)
            self.assertIn(f'data-page="{page}"', self.page)
        self.assertIn("function activatePage", self.page)
        self.assertIn("hashchange", self.page)

    def test_settings_page_exposes_live_readiness_assessment_without_enable_button(self):
        self.assertIn('id="runLiveReadiness"', self.page)
        self.assertIn('id="liveReadinessOutput"', self.page)
        self.assertIn("/live-readiness", self.page)
        self.assertNotIn("启用实盘", self.page)

    def test_settings_page_exposes_secure_tushare_token_configuration(self):
        for element_id in ("tushareToken", "saveTushareToken", "tushareTokenStatus"):
            self.assertIn(f'id="{element_id}"', self.page)
        self.assertIn('type="password"', self.page)
        self.assertIn("/settings/tushare-token", self.page)
        self.assertNotIn("value=\"dd804", self.page)

    def test_model_answer_is_not_interpolated_into_inner_html(self):
        self.assertNotIn("${result.answer}${plan}${evidence}", self.page)
        self.assertIn("function renderAgentResult", self.page)
        self.assertIn("answer.textContent = result.answer", self.page)
        self.assertIn("300000", self.page)
        self.assertIn("最长等待 300 秒", self.page)

    def test_agent_exposes_stock_research_template_and_asset_fields(self):
        for element_id in (
            "agentTemplate",
            "agentAssetCode",
            "agentAssetName",
            "applyAgentTemplate",
        ):
            self.assertIn(f'id="{element_id}"', self.page)
        self.assertIn('value="a-share-fundamental"', self.page)
        self.assertIn('value="etf-tushare-free"', self.page)
        self.assertIn('value="a-share-daily-free"', self.page)
        self.assertIn("function buildStockResearchPrompt", self.page)
        self.assertIn("function buildEtfResearchPrompt", self.page)
        self.assertIn("function buildFreeStockDailyPrompt", self.page)
        self.assertIn("function currentResearchDate", self.page)
        self.assertIn("研究截止日：${currentResearchDate()}", self.page)
        self.assertIn("工具返回真实数据，模型解释仍需核验", self.page)
        self.assertNotIn("工具已执行：有真实数据证据", self.page)
        self.assertIn("003019.SZ", self.page)
        self.assertIn("不得凭记忆", self.page)
        self.assertIn("fund_daily", self.page)
        self.assertIn("510300.SH", self.page)

    def test_all_browser_posts_use_csrf_protected_fetch(self):
        self.assertIn("function securePostFetch", self.page)
        self.assertIn("/csrf-token", self.page)
        self.assertIn("X-CSRF-Token", self.page)
        self.assertNotIn("fetch('/service/restart', {method:'POST'", self.page)
        self.assertNotIn("fetch('/agent/research', {method:'POST'", self.page)


if __name__ == "__main__":
    unittest.main()
