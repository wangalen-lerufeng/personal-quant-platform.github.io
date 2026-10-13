# MVP 实现计划

## Task 1：项目基础与领域模型

先为 `MarketEvent`、`Signal`、`OrderIntent`、`RiskDecision`、`OrderStatus` 编写失败测试，再实现 Pydantic 模型和枚举。

## Task 2：执行网关契约

定义统一的 `ExecutionGateway` 协议，包含 `submit_order`、`cancel_order`、`get_positions`、`health_check`。先写契约测试，再实现内存模拟网关。

## Task 3：三市场适配器骨架

创建 QMT、CTP、MT5 三个适配器目录。第一阶段只实现模拟行情和模拟订单，不安装或调用真实 SDK。

## Task 4：风控引擎

实现订单数量、单笔风险、单日亏损、最大持仓、交易时段和 Kill Switch 规则，并确保所有订单必须先经过风控。

## Task 5：订单状态机与审计

实现订单状态转换、幂等键、拒单原因、成交回报和 append-only 审计事件。

## Task 6：策略运行时

实现一个不依赖市场适配器的示例趋势策略，输出标准 `Signal`，由组合模块转换为 `OrderIntent`。

## Task 7：AI Agent 研究工具

只接入回测、行情查询和复盘工具；暂不提供任何执行工具。所有工具输入输出使用结构化模型。

## Task 8：最小管理 API 与验证

提供健康检查、策略列表、风控状态和模拟订单接口，运行完整测试与静态检查。
