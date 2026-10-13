# 验证与安全边界

## 本地验证命令

```bash
cd /Users/apple/personal-quant-platform
PYTHONPATH=src python3 -m unittest discover -s tests -q
PYTHONPATH=src python3 -m compileall -q src tests examples
PYTHONPATH=src python3 -m quant_platform check
PYTHONPATH=src python3 examples/run_paper_loop.py
```

启动只读控制台：

```bash
PYTHONPATH=src python3 -m quant_platform serve --host 127.0.0.1 --port 8080
```

浏览器打开 `http://127.0.0.1:8080/`。页面提供只读研究接口和隔离的 paper-only 仿真接口，不提供真实下单能力。

访达启动脚本通过用户级 `launchd` 托管服务。验证控制台重启时，`POST /service/restart` 应返回 `supervisor: launchd`，随后 `/health` 在 2–5 秒内恢复；终端退出不应影响服务。

2026-10-12 验证：控制台重启后服务恢复，后台进程由 `launchd` 接管；启动包装器使用项目专用的 `QP_SERVICE_HOST=127.0.0.1`，避免用户 shell 中的通用 `HOST` 变量覆盖本地绑定。

2026-10-13 验证：本地 POST 来源验证已补齐 Referer 兜底。此前仅带 `Referer`、不带 `Origin` 的浏览器请求会提前返回而跳过 CSRF；现改为 `Origin` 或 `Referer` 任一存在即校验来源并强制 CSRF token，跨域/跨协议来源一律拒绝。全量 273 项测试与编译检查通过，`quant_platform check` 确认所有适配器仍为 paper 模式。

安全回归还应确认：模型回答只以纯文本写入页面；AI 工具拒绝模型传入 `token_env` 等凭据选择器；服务拒绝绑定非回环地址；数据库、日志和目录权限仅授予当前用户；运行依赖从 `requirements.lock` 安装。

研究行情查询可使用 `GET /market-data?symbol=EURUSD&limit=100` 获取指定品种的最新 N 条记录；该接口仅返回数据，不提供下单或撤单能力。

控制台的“运行纸面回测”会调用 `POST /backtest`。接口只接受历史行情与回测参数，响应带有 `research_only: true` 和 `execution_enabled: false`，不会进入交易管线。

启用市场规则后，事件回测会过滤非交易时段，按显式前收盘价和涨跌幅限制拦截涨停买入/跌停卖出，并在换月生效后拒绝到期合约的新信号。所有拦截记录在 `restrictions`；当前不会自动迁移已有期货持仓。

`POST /factor-panel-analysis` 只使用调用者提供的显式未来收益标签，按日期计算横截面 IC/Rank IC，并报告 ICIR、衰减、自相关、换手率、容量估算和样本外拆分。容量是流动性启发式估算，不能视为实盘成交承诺。

## 当前安全不变量

- `live_trading_enabled` 必须为 `false`。
- `ai_execution_tools_enabled` 必须为 `false`。
- 三个适配器均为 `paper` 模式。
- 健康、控制面、研究、行情、组合和审计接口均只读；仿真接口只修改本地虚拟账户。所有接口返回 `execution_enabled: false`。
- AI 研究工具可以查询行情、组合、审计和回测结果，但目录中没有下单工具。
- 订单必须先经过风险引擎；适配器只负责市场隔离和 paper 契约。
- 在线行情客户端只读，API 密钥通过环境变量注入；它不具备下单、撤单或修改账户状态的方法。
- 本地 POST 接口在分发前统一经过来源验证：Host 必须回环；Origin/Referer（任一存在）必须为 http 且与 Host 同源并回环；任一浏览器来源存在即强制 CSRF token。无 Origin/Referer 的本机 CLI JSON 调用仍受支持，但不会获得 CSRF 豁免。
- 仿真会话同时检查账户回撤、连续亏损、行情陈旧、API 连续失败和人工停止状态。
- 仿真快照和会话状态写入配置的 SQLite 数据库；现金、持仓、未成交订单、风险计数和人工停止状态均须在重启后恢复。
- 纸面成交证据随会话状态持久化；`/simulation/reconciliation` 从成交重建现金、持仓和订单状态，`/simulation/acceptance` 要求至少 30 个不同日期的快照且无风控阻断。通过后仍不会开启实盘。
- 已完成本机双重启冒烟验证：部分成交订单与人工停止状态在第二个服务进程中保持一致。
- `POST /live-readiness` 只评估样本外测试、仿真时长、订单对账、风险限制和人工审批证据；不会更改运行配置或开启实盘。
- 行情 REST 客户端只进行有限次数重试；每次失败/恢复可通过健康回调进入 API 故障保护。
- 高级因子分析不在内部生成未来标签，并通过接口字段标记 `future_data_used_for_feature: false`；标签生成仍须由数据管线另行审计。

## 当前未接入

QMT/PTrade、CTP 原生 SDK、MetaTrader 5 SDK、真实券商账户和真实订单执行均未接入。接入前必须增加模拟盘、对账、断线重连和人工审批验证。

## DeepSeek 调用验证

设置 `AI_API_KEY` 并启用 `model` 配置后，使用 `--config` 指定 YAML 文件重启服务：`PYTHONPATH=src python3 -m quant_platform serve --config configs/settings.example.yaml --host 127.0.0.1 --port 8080`。不指定 `--config` 时 CLI 使用默认配置，模型会显示未启用。AI 研究助手显示“DeepSeek 研究完成”及模型回答，才算调用成功；“编排完成”是本地规则模式。`Unexpected end of JSON input` 通常表示页面连接到了已停止或旧版本服务。当前项目未持久化模型调用日志，不能对历史请求做事后确认。

若出现 `CERTIFICATE_VERIFY_FAILED`，安装项目依赖（包含 `certifi`）后重启服务；不要使用关闭证书校验的方式绕过问题。

macOS Apple Silicon 注意：Python、NumPy、Pandas 和 AkShare 必须使用同一架构。若出现 `incompatible architecture (have arm64, need x86_64)`，使用 `arch -arm64 /usr/local/bin/python3` 启动和安装依赖。

LightGBM（Phase 9 step2 基线）使用原生 API，不依赖 scikit-learn。其 arm64 wheel 需要 OpenMP 运行时 `libomp.dylib`；若本机 Homebrew 只有 x86_64 libomp，可在 lightgbm 包目录放置 arm64 `libomp.dylib` 并用 `install_name_tool -change @rpath/libomp.dylib @loader_path/libomp.dylib` 改写依赖，使其自包含。重装 lightgbm 后需重做该步骤。

AkShare 1.18.x 的 `forex_hist_em` 只接受 `symbol` 参数，项目会在本地按 `start_date` / `end_date` 过滤，不再把日期参数直接传给 AkShare。
