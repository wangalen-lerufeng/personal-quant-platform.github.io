# Personal Quant Platform

面向中国大陆个人开发者的多市场量化交易平台骨架。

## 目标

统一研究、策略、AI、风控、数据和监控能力，并通过可插拔执行适配器支持：

- A 股 / ETF：预留 QMT、PTrade 接口
- 国内期货：预留 CTP 接口
- 海外外汇 / 贵金属：预留 MT5 接口

第一阶段只建设平台核心与适配器契约，不连接真实账户，不执行真实订单。

当前已具备：

- 市场无关的领域模型
- QMT、CTP、MT5 三个纸面执行适配器
- `AdapterRegistry` 统一路由三市场行情与 paper 订单
- 统一风控到执行的交易编排管线
- 订单生命周期和审计事件骨架
- 部分成交、成交数量约束与追加式状态审计
- 支持卖出信号平仓的回测与权益曲线
- 407 项标准库单元测试
- 只读 AI 研究工具与回测实验查询
- 可运行的纸面交易示例：`examples/run_paper_loop.py`

## 当前阶段

本项目处于纸面交易 MVP 阶段。当前核心链路可在无真实账户的情况下运行；真实 SDK 仍未接入。详细设计见：

- `docs/plans/2026-10-08-platform-design.md`
- `docs/plans/2026-10-08-mvp-implementation-plan.md`
- 验证与安全边界：`docs/verification.md`

行情研究接口支持 `GET /market-data?symbol=EURUSD&limit=100`，其中 `limit` 返回该品种最新的 N 条标准化行情记录。

## 安全边界

任何真实交易前必须完成模拟盘、风控验证、订单对账、断线重连和人工审批流程。AI Agent 不直接拥有券商或交易所密钥，也不能绕过风险引擎下单。

## 功能能力

本项目为纯纸面（paper-only）量化研究平台，真实交易全程关闭。当前能力如下：

### 数据获取（Tushare 在线 + 本地）

| 资产类型 | 数据源 | 复权 |
|---|---|---|
| A 股 ETF | `fund_daily` + `fund_adj` | ✅ |
| A 股个股 | `daily` + `adj_factor` | ✅ |
| 港股 | `hk_daily` | 暂未复权 |
| 指数成分 | `index_weight`（沪深300 / 中证500 / 上证50 自动取成分） | 走对应个股/ETF |
| 关键词转股票 | AnySearch `/v1/search`（关键词 → A 股代码，需在系统设置配置 API Key） | — |
| 本地 CSV | `timestamp,symbol,asset_class,price` | — |

### 因子与特征

- 基础特征：收益率、动量、波动率、ATR、均线偏离、成交量比率。
- 因子分析：IC、Rank IC、分组收益、ICIR、因子衰减、自相关、换手率、容量估计、样本内外拆分。
- Qlib Alpha158：158 个横截面因子。
- 自定义因子：Qlib 表达式引擎（`Ref`/`Mean`/`Std`/`Corr`/`Rank` 等），控制台输入框直接填写。

### 模型

| 模型 | 说明 |
|---|---|
| LightGBM | 原生 API + Qlib `LGBModel` |
| DoubleEnsemble | 自研版 + Qlib `DEnsembleModel`（样本重加权 + 特征选择） |
| 约束 PPO | 纯 numpy 迷你 PPO（仓位 0.8–1.2 倍微调） |

### 组合研究流水线（核心）

数据 → Alpha158/自定义因子 → LGBModel/DEnsembleModel 候选池 → TradingAgents 验真 → 约束 PPO 定仓 → 确定性风控 → 纸面下单 → DeepSeek 大模型解读。

### 交易（全部纸面，不接真实券商）

- 纸面账户：创建、下单、撮合、部分成交、账户估值、快照持久化。
- 风控：Kill Switch、单日亏损、最大回撤、单笔/持仓/金额上限、连续亏损、行情陈旧、API 失败、全局熔断。
- 自动对账：从成交证据重建现金/持仓/订单并报告差异。
- 30 日验收闸门：满 30 个仿真交易日 + 对账通过 + 无风控阻断，才允许申请人工实盘评估（实盘仍关闭）。

### 回测

事件驱动回测：手续费、滑点、交易时段、涨跌停、期货换月、样本内外划分。

### 前向验证（纸面对比实盘）

选股后记录 Day-0 持仓与基准点位，之后每天用真实 Tushare 收盘价做 mark-to-market，对比策略组合与基准指数（默认沪深 300）的累计收益、超额收益与回撤，并逐只标注是否跑赢基准。仅研究，不产生订单。

### 运行历史 / 复盘

每次「组合研究流水线」和「前向验证」都会存档到本地 `data/run_history.json`：组合研究记录候选池、目标仓位、模型 Rank IC、下单笔数与对应仿真账户；前向验证记录基准、持仓与逐日收益，并可追溯到来源的组合研究 run_id。控制台「运行历史 / 复盘」卡片可随时刷新查看。

### AI（只读，无执行权限）

- AI 研究助手：DeepSeek + Tushare 个股深度研究（基本面/技术面/新闻）。
- 组合解读：流水线跑完后自动生成投资解读。
- 所有 AI 工具走只读白名单，不能下单、改仓、改风控。

### 控制台（Web，8 页）

总览、数据中心、研究中心、快速回测、组合研究、仿真交易、风险监控、系统设置。

### 明确不支持

- 真实下单/撤单（QMT/PTrade/CTP/MT5 真实 SDK 未接入）。
- 保证盈利。
- AI 直接交易。

## 本地运行

```bash
cd /Users/apple/personal-quant-platform
PYTHONPATH=src python3 -m unittest discover -s tests -q
PYTHONPATH=src python3 -m quant_platform check
PYTHONPATH=src python3 -m quant_platform serve --config configs/settings.example.yaml --host 127.0.0.1 --port 8080
PYTHONPATH=src python3 examples/run_paper_loop.py
PYTHONPATH=src python3 examples/run_csv_backtest.py /path/to/history.csv --commission-rate 0.0003 --slippage-rate 0.0001
PYTHONPATH=src arch -arm64 /usr/local/bin/python3 examples/run_phase9_pipeline.py
```

Phase 9 组合研究流水线示例（`examples/run_phase9_pipeline.py`）在合成 ETF 面板上跑通完整链路：数据地基 → LightGBM/DoubleEnsemble 候选池 → TradingAgents 证据验真 → 约束 PPO 定仓 → 确定性风控 → 纸面仿真，并打印各阶段结果与审计记录。该示例依赖 LightGBM（arm64），需用 `arch -arm64` 运行（见 `docs/verification.md`）。

启动服务后，在浏览器打开 `http://127.0.0.1:8080/`，即可使用浅色控制台查看系统、风控、组合、适配器和只读行情。

访达双击 `启动量化平台.command` 后，服务由 macOS `launchd` 在后台托管；终端窗口可以关闭。控制台“重启服务”会退出当前进程，再由 `launchd` 自动拉起，不会丢失 SQLite 中的仿真状态。环境变量仍由 zsh 包装脚本读取 `~/.zshrc`，不会再用 Bash 直接加载 zsh 配置。

控制面接口：

```text
GET /health       系统健康状态
GET /system       运行环境与安全开关
GET /risk         风控状态
GET /adapters     QMT / CTP / MT5 适配器状态
GET /orders       订单审计记录
GET /audit-events 追加式订单状态事件，可用 order_id 筛选
GET /experiments  回测实验结果
GET /strategies   已登记策略元数据（研究只读）
GET /research     只读研究工具目录（不提供下单能力）
GET /portfolio    只读组合估值快照
GET /market-data  只读行情查询，可用 symbol 参数筛选
GET /research/portfolio  只读研究组合快照
POST /backtest       研究回测接口（JSON 行情输入，只返回指标，不执行订单）
POST /order-features 订单流特征分析（买卖不平衡、VWAP、大单和价格冲击）
POST /simulation-risk 仿真权益风险回放（单日亏损、回撤与熔断告警）
POST /factor-analysis 因子 IC、Rank IC 与分组收益分析
POST /factor-panel-analysis 按日 IC/ICIR、衰减、自相关、换手率、容量与样本外分析
POST /market-state   可解释的趋势、震荡和高波动状态识别
POST /simulation/create  创建隔离的 paper-only 仿真账户
POST /simulation/order   提交仿真订单（不连接真实网关）
POST /simulation/market  推进市场事件、撮合并保存账户快照
POST /simulation/control 记录盈亏/API 状态或人工停止仿真
POST /simulation/reconciliation 从成交证据重建现金、持仓和订单状态并报告差异
POST /simulation/acceptance 统计仿真交易日并执行 30 日验收闸门
GET  /simulation/state?account_id=paper-main  查询仿真账户与风险状态
POST /live-readiness  评估实盘申请证据；只允许进入人工审核，不开启实盘
```

`POST /backtest` 可选 `enforce_market_rules: true`，用于启用交易时段、涨跌停和期货换月拦截。响应中的 `restrictions` 会记录时间、品种、阶段和原因。涨跌停与前收盘价需要通过 `previous_closes`、`price_limit_percentages` 显式提供；期货换月通过 `rollover_rules` 提供。默认只阻止到期合约继续产生新交易；研究回测可额外显式设置 `auto_rollover: true`，在生效时间后的后续合约行情到达时，以旧合约最后行情平仓、按同等数量建立后续合约仓位。响应中的 `rollovers` 会记录两侧价格、数量、手续费和滑点。该能力仅属于 paper-only 回测，不会调用 CTP 或产生实盘订单。

高级因子接口只接受名称中含 `forward` 的显式未来收益标签，不在分析器内部生成未来收益。容量结果采用“头部分组每日成交额中位数 × 参与率”的研究估算，不等同于可成交资金规模。

仿真交易台使用 `storage.database` 指定的 SQLite 数据库保存账户快照及活动会话。示例配置为 `data/platform.db`。服务重启后会恢复现金、持仓、最新价格、部分成交/未成交订单、账户风险、连续亏损/API 故障计数和人工停止状态。

新产生的纸面成交也会写入会话快照，用于自动对账。验收闸门要求至少 30 个不同日期的账户快照、自动对账无差异且账户/运行风控未停机；通过后也只会返回“可进入人工审核”，不会打开实盘。

控制台使用 URL 哈希切换总览、数据中心、研究中心、快速回测、仿真交易、风险监控和系统设置。行情 REST 客户端支持有限次数重试及健康回调，可将持续故障计入仿真运行风险；不会无限重试。

## 历史行情 CSV

风险配置使用绝对数量/金额字段，例如 `max_order_quantity`、`max_position_quantity`、`max_order_value`、`max_daily_loss` 和 `max_drawdown`；比例字段不会被自动换算。

回测加载器使用以下最小格式：

```csv
timestamp,symbol,asset_class,price
2026-01-01T00:00:00+00:00,EURUSD,fx,1.08
```

支持的 `asset_class`：`fx`、`etf`、`futures`。

## DeepSeek 接入

```yaml
model:
  enabled: true
  provider: deepseek
  base_url: https://api.deepseek.com
  name: deepseek-flash
  api_key_env: AI_API_KEY
  temperature: 0.2
  max_tokens: 2000
```

```bash
export AI_API_KEY="你的 DeepSeek API Key"
PYTHONPATH=src python3 -m quant_platform serve --host 127.0.0.1 --port 8080
```

启动时必须使用 `--config configs/settings.example.yaml`（或你的配置文件路径），否则 CLI 会使用默认配置，页面显示“模型未启用”。打开 `http://127.0.0.1:8080/` 后，在“AI 研究助手”中提交问题。显示“DeepSeek 研究完成”并出现模型回答，才表示本次调用成功；显示“编排完成”表示使用本地规则编排。项目当前不保存模型调用日志，因此无法事后确认已经结束的请求。

如遇 `CERTIFICATE_VERIFY_FAILED`，项目会使用 `certifi` 根证书，不要关闭 SSL 校验。生产运行环境优先使用已审核的锁定版本安装：`python3 -m pip install -r requirements.lock`，开发模式再执行 `python3 -m pip install -e . --no-deps`。升级依赖前必须重新运行测试与安全检查。

## 在线行情 API

系统提供只读 `RestMarketDataClient`，通过配置文件指定行情服务地址，密钥只从环境变量读取：

```bash
export MARKET_DATA_API_KEY='your-token'
```

配置 `configs/settings.example.yaml` 中的 `market_data.url`、`api_key_env` 和 `timeout_seconds`。供应商接口需要返回 `data` 数组（或直接返回数组），每条记录包含 `symbol`、`price` 和带时区的 `timestamp`。Tushare、AkShare、MT5 等供应商应分别编写薄适配器，将返回字段转换为统一格式。

桌面后台服务使用 Tushare 官方 SDK 默认地址，不启用第三方兼容服务或带 Token 的 MCP URL。令牌可在“系统设置”页面保存到本机 `secrets/tushare.env`；目录权限为 `700`、文件权限为 `600`，并已被 Git 忽略。API 和页面只返回是否配置，永不回显令牌。
