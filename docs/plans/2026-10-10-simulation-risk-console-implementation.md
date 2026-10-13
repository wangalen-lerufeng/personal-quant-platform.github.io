# 仿真账户、运行保护与控制台实施计划

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** 将纸面撮合、账户估值、快照、风险保护和控制台联成可追溯的仿真闭环，同时保持真实交易与 AI 执行权限关闭。

**Architecture:** 新增一个只负责 paper-only 仿真的会话协调器，组合现有 `PaperMatchingEngine`、`RiskMonitor` 和 `SimulationAccountStore`。运行保护器独立跟踪连续亏损、行情新鲜度和外部 API 故障；HTTP 层只暴露创建、推进、查询仿真会话的本地接口，所有响应明确携带 `execution_enabled: false`。

**Tech Stack:** Python 3.11、SQLite、标准库 HTTPServer、pandas、unittest、原生 HTML/CSS/JavaScript。

---

### Task 1: 仿真会话协调器

**Files:**
- Create: `src/quant_platform/simulation.py`
- Create: `tests/test_simulation_session.py`
- Modify: `src/quant_platform/execution.py`

**Steps:**
1. 先写失败测试，覆盖提交订单、部分成交、现金/持仓/权益更新和快照持久化。
2. 运行 `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_simulation_session.py' -q`，确认因模块缺失而失败。
3. 实现最小 `PaperSimulationSession`，只接受 paper 撮合，不接真实网关。
4. 重跑目标测试并确认通过。

### Task 2: 连续亏损与运行健康保护

**Files:**
- Modify: `src/quant_platform/risk.py`
- Modify: `tests/test_simulation_session.py`

**Steps:**
1. 先写失败测试，覆盖连续亏损阈值、行情超时、API 连续故障和手动熔断。
2. 确认测试按预期失败。
3. 实现独立 `OperationalRiskGuard`，仅产生告警和暂停状态，不拥有下单能力。
4. 运行目标测试及现有风险测试。

### Task 3: 仿真 HTTP 接口

**Files:**
- Modify: `src/quant_platform/http_api.py`
- Modify: `tests/test_http_api.py`

**Steps:**
1. 先写失败测试，覆盖创建仿真、提交纸面订单、推进市场事件和读取状态。
2. 实现进程内仿真会话注册表和 `/simulation/*` 接口。
3. 验证所有响应包含 `paper_only: true` 或 `execution_enabled: false`。
4. 运行 HTTP 测试与完整测试。

### Task 4: 控制台仿真页面

**Files:**
- Modify: `web/index.html`
- Modify: `tests/test_web_ui.py`

**Steps:**
1. 先写失败静态页面测试，锁定仿真输入、按钮、状态与告警区域。
2. 增加 paper-only 仿真卡片和调用逻辑。
3. 确认页面不包含实盘下单入口，并保持全局停止提示可见。
4. 运行页面测试和完整测试。

### Task 5: 文档与发布闸门

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/verification.md`

**Steps:**
1. 记录仿真流程、接口、已知限制和安全边界。
2. 运行 `PYTHONPATH=src python3 -m unittest discover -s tests -q`。
3. 运行 `PYTHONPATH=src python3 -m compileall -q src`。
4. 本机重启服务，冒烟验证健康状态、仿真闭环和安全开关。

### Task 6: 实盘申请资格闸门

**Files:**
- Create: `src/quant_platform/live_readiness.py`
- Create: `tests/test_live_readiness.py`
- Modify: `src/quant_platform/http_api.py`

**Steps:**
1. 先写失败测试，覆盖样本外测试、仿真天数、订单对账、风控配置和人工审批证据。
2. 实现只读资格评估；满足条件只允许 `eligible_to_request: true`，不得改变运行配置。
3. 暴露 `POST /live-readiness` 研究接口，响应始终包含 `execution_enabled: false`。
4. 运行目标测试、HTTP 测试和完整回归。

### Task 7: 高级因子与事件回测市场规则

**Files:**
- Modify: `src/quant_platform/factor_analysis.py`
- Modify: `src/quant_platform/backtest.py`
- Modify: `src/quant_platform/http_api.py`
- Modify: `web/index.html`

**Steps:**
1. 先写失败测试，覆盖按日 IC/ICIR、衰减、自相关、换手率、容量和样本外拆分。
2. 要求显式未来收益标签，不在分析器中构造未来数据。
3. 先写失败测试，再实现交易时段、涨跌停与过期合约换月拦截。
4. 将高级因子与市场规则参数接入控制台，保持所有执行开关关闭。
5. 运行完整回归、编译检查与本机接口冒烟验证。
