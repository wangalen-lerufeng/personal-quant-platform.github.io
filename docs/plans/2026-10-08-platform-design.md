# Personal Quant Platform 统一平台设计

## 1. 目标与非目标

### 目标

构建一个统一的 Python 量化平台核心，使同一套策略、组合管理、风险规则和审计能力能够通过不同适配器连接 A 股/ETF、国内期货和 MT5 海外品种。

### 非目标

- 第一阶段不接入真实券商账户
- 不实现高频交易或低延迟 C++ 执行
- 不允许 AI Agent 自主修改硬风控或直接下单
- 不把 MT5 当作 A 股或国内期货的执行通道

## 2. 总体架构

```text
行情适配器 -> MarketEvent -> Feature Engine -> Strategy Engine
                                             |
                                             v
                                      Signal / Target Position
                                             |
                                             v
                                      Risk Engine
                                             |
                                             v
                              Order Intent / Execution Gateway
                               |              |             |
                         QMT/PTrade         CTP           MT5
```

AI Agent 位于研究与控制面：通过工具调用行情查询、回测、策略分析和复盘；不能绕过 Risk Engine 直接调用执行适配器。

## 3. 核心模块

### Domain

定义市场无关的数据契约：`MarketEvent`、`Bar`、`Signal`、`OrderIntent`、`OrderStatus`、`Position`、`AccountSnapshot`、`RiskDecision`。

### Market Data

统一行情事件、交易日历、品种元数据和数据质量标记。适配器负责把不同市场原始数据转换为平台标准模型。

### Strategy

策略只依赖标准行情和组合状态，不依赖 QMT、CTP 或 MT5 SDK。策略输出信号或目标仓位，不直接下单。

### Risk

至少支持单笔风险、单日亏损、最大回撤、最大持仓、最大敞口、保证金、交易时段、价格偏离和 Kill Switch。

### Execution

通过 `ExecutionGateway` 管理订单提交、撤单、改单、重试、幂等、成交回报和对账。三个适配器均实现同一接口。

### AI Agent

第一阶段只提供研究助手能力：回测、解释、实验记录、交易复盘和告警摘要。Agent 的工具返回结构化结果，所有策略变更必须经过版本登记和人工审批。

### Persistence

第一阶段使用 PostgreSQL 存储业务、策略和审计数据；行情原始文件使用 Parquet。后续根据数据量增加 TimescaleDB、ClickHouse 或对象存储。

## 4. 适配器边界

| 适配器 | 目标市场 | 第一阶段职责 |
|---|---|---|
| `qmt` | A 股 / ETF | 定义接口与模拟实现 |
| `ctp` | 国内期货 | 定义接口与模拟实现 |
| `mt5` | EURUSD、XAUUSD 等 MT5 品种 | 定义接口与模拟实现 |

真实 SDK 必须放在适配器内部，核心模块不能导入 `MetaTrader5`、`xtquant` 或 CTP 原生包。

## 5. 第一阶段完成标准

- `Signal -> RiskDecision -> OrderIntent` 可以在无真实账户的情况下跑通
- 三个执行适配器都能通过同一组契约测试
- 风控拒单、熔断和重复订单保护有测试
- 订单状态机能够处理提交、确认、部分成交、完全成交、撤单和拒单
- AI Agent 只能调用研究工具，无法直接调用执行接口
- 配置、日志和审计事件可复现

## 6. 推荐技术

- Python 3.12
- FastAPI：管理接口
- Pydantic：领域数据模型与配置校验
- Polars / NumPy：数据与特征
- PostgreSQL：业务、策略、订单和审计
- Redis：实时状态和事件
- pytest：测试
- Docker Compose：本地依赖环境
- LangGraph：后续 AI Agent 编排
