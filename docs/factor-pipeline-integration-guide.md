# 个人量化平台：因子处理与 LightGBM 集成指南

> 适用项目：`personal-quant-platform`  
> 研究范围：Qlib Alpha158/自定义因子、LightGBM、DoubleEnsemble、Phase 9 组合研究  
> 安全原则：仅用于研究与纸面交易；回测收益不代表未来收益，不构成投资建议。

## 1. 核心结论

基于当前系统，缺失值检查、去极值、截面标准化、行业和市值中性化以及因子相关性聚类，应放在 **Qlib 计算原始因子之后、LightGBM/DoubleEnsemble 训练之前**。

样本外 Rank IC、分组收益单调性、换手与交易成本、市场状态分组测试，应放在 **模型输出预测分数之后、生成候选池之前**。

最终应增加模型准入门：模型不达标时保持现金，不生成候选池、目标仓位或订单。

## 2. 当前系统数据流

```text
Tushare 行情
  ↓
复权、停牌过滤、流动性过滤、前向收益标签
  ↓
转换成 Qlib 数据
  ↓
Alpha158 或自定义表达式计算因子
  ↓
LightGBM / DoubleEnsemble 训练
  ↓
样本外预测分数
  ↓
候选池
  ↓
TradingAgents 验真
  ↓
PPO 仓位微调
  ↓
确定性风控
  ↓
纸面交易
```

建议调整为：

```text
Tushare 行情
  ↓
数据质量与可交易性过滤
  ↓
Qlib 计算原始因子
  ↓
【因子处理流水线】
  1. 缺失值检查
  2. 截面去极值
  3. 行业和市值中性化
  4. 截面标准化
  5. 因子相关性聚类
  ↓
LightGBM / DoubleEnsemble
  ↓
【模型样本外评估】
  1. 样本外 Rank IC / ICIR
  2. 分组收益单调性
  3. 换手与交易成本
  4. 市场状态分组
  ↓
【模型准入门】
  ├─ 未达标：保持现金，不生成订单
  └─ 达标：候选池 → 验真 → PPO → 风控 → 纸面交易
```

## 3. 功能现状与目标位置

| 功能 | 当前状态 | 正确位置 |
|---|---|---|
| 缺失值检查 | 部分存在，不完整 | Qlib 因子计算后 |
| 1%/99% 去极值 | 有基础函数，未接入 Phase 9 | LightGBM 之前 |
| MAD 去极值 | 尚未实现 | LightGBM 之前 |
| 截面标准化 | 有基础函数，未接入 Phase 9 | LightGBM 之前 |
| 行业、市值中性化 | 尚未实现 | LightGBM 之前 |
| 因子相关性聚类 | 尚未实现 | LightGBM 之前 |
| 单因子 Rank IC | 已实现，但目前使用全区间 | 因子筛选阶段 |
| 模型样本外 Rank IC | 已实现 | LightGBM 之后 |
| 分组收益 | 分析模块已支持，未完整接入准入流程 | 模型之后 |
| 换手率 | 已计算并展示 | 模型之后 |
| 真实交易成本 | 回测模块有，Phase 9 未完整接入 | 模型之后 |
| 市场状态分类 | 独立模块已实现，未接入 Phase 9 | 模型之后 |
| 模型不合格时停止下单 | 尚未实现 | 候选池之前 |

## 4. 数据质量处理

当前数据地基位于：

```text
src/quant_platform/etf_universe.py
```

现有职责：

- 合并复权因子；
- 计算复权收盘价；
- 排除零成交量或停牌记录；
- 流动性过滤；
- 生成前向收益标签；
- 日期排序；
- 去除重复记录。

建议在该阶段补充：

- OHLCV 字段缺失；
- 非正价格；
- 成交量、成交额异常；
- 复权因子缺失；
- 每只资产历史长度不足；
- 股票池历史成分有效性；
- 前向标签末端自然缺失。

因子值本身的缺失只能在 Qlib 完成因子计算后检查。

## 5. 因子处理流水线

当前 Qlib 入口位于：

```text
src/quant_platform/qlib_alpha158.py
```

关键入口：

```text
_make_custom_handler()
run_alpha158()
```

建议增加独立模块：

```text
src/quant_platform/factor_preprocessing.py
```

目标职责：

```text
原始 Qlib 因子面板
→ 缺失率报告
→ 截面 MAD/分位数去极值
→ 行业和市值中性化
→ 截面标准化
→ 输出处理后的训练面板
```

建议接口：

```python
processed = preprocess_factor_panel(
    raw_factors,
    train_start=train_start,
    train_end=train_end,
    winsor_method="mad",
    standardize="cross_sectional_zscore",
    neutralize=("industry", "log_market_cap"),
)
```

### 5.1 缺失值

建议记录：

- 每个因子的缺失比例；
- 每个日期的有效股票数；
- 每只股票的有效历史长度；
- 缺失原因；
- 填充或删除规则。

注意事项：

- 标签缺失样本应删除；
- 新股历史不足不能简单填零；
- 不能使用全样本均值填充训练数据；
- 所有需要拟合的填充参数只能在训练期计算。

### 5.2 去极值

当前基础函数位于：

```text
src/quant_platform/factor_analysis.py
```

其中 `winsorize_zscore()` 已支持分位数截断和 Z-Score，但尚未接入 Phase 9 模型训练。

正式实现必须按交易日横截面执行：

```python
panel.groupby("timestamp")[factor].transform(winsorize_zscore)
```

不能把不同年份全部混合后计算统一分位点。

建议同时支持 MAD：

```text
中位数 ± 3～5 × 1.4826 × MAD
```

### 5.3 行业和市值中性化

推荐顺序：

```text
原始因子
→ 去极值
→ 行业和市值中性化
→ 截面 Z-Score
```

每日横截面回归：

```text
因子值 = 行业哑变量 + log(总市值) + 残差
```

模型使用残差，避免把行业或小市值暴露误认为 Alpha。

当前 Phase 9 面板尚缺少：

```text
industry
total_mv / circ_mv
```

需要从 Tushare 补充行业分类和 `daily_basic` 市值字段。

ETF 不适合直接使用上市公司行业中性化，可改用：

- ETF 类型；
- 跟踪指数类别；
- 行业 ETF 分类；
- 宽基、行业、主题、债券、商品等资产类别。

### 5.4 因子相关性聚类

建议新增：

```text
src/quant_platform/factor_selection.py
```

流程：

```text
训练期清洗后因子
→ Spearman 相关性矩阵
→ 按 |相关系数| 聚类
→ 每簇保留代表因子
→ 输出最终特征清单
```

代表因子选择依据：

- 训练期和验证期 IC 稳定性；
- 缺失率；
- 换手率；
- 经济含义；
- 计算复杂度；
- 与其他因子的独立性。

同周期的 `VSUMP_N`、`VSUMN_N`、`VSUMD_N` 几乎为确定性线性变换，应只保留一个。

## 6. 单因子诊断

当前实现位于：

```text
src/quant_platform/qlib_alpha158.py
```

函数：

```text
compute_factor_ic_report()
```

现有输出：

- 日度 Rank IC；
- Rank IC 均值；
- ICIR；
- `keep/drop`；
- 推荐因子列表。

当前主要问题：因子诊断使用整个开始日至结束日进行筛选，最终测试期也参与了因子选择，存在数据窥探风险。

应调整为：

```text
训练期：计算 IC、相关性并筛选因子
验证期：确定阈值和参数
测试期：仅做最终评价，不再修改因子
```

网页中的“应用建议因子”只能作为探索性研究功能，不能把全区间推荐结果视为正式样本外结论。

## 7. LightGBM 与 DoubleEnsemble

模型入口位于：

```text
src/quant_platform/qlib_alpha158.py
src/quant_platform/qlib_model.py
```

当前 `run_alpha158()` 按时间切分：

```text
前 70%：训练
中间 15%：验证
最后 15%：测试
```

模型只在测试段输出预测，方向上属于样本外评估。

仍需补充：

- 根据标签持有期设置切分隔离区；
- Walk-forward 滚动训练；
- 多个时间窗口稳定性；
- 随机种子重复试验；
- 模型与数据版本记录。

## 8. 模型输出后的评估

当前相关模块：

```text
src/quant_platform/qlib_model.py
src/quant_platform/factor_analysis.py
src/quant_platform/report_charts.py
```

已有能力：

- 样本外预测分数；
- Rank IC；
- ICIR；
- 候选池；
- IC 时间序列；
- 换手率图表；
- 收益和回撤图表。

### 8.1 样本外 Rank IC

建议同时展示：

- Rank IC 均值；
- Rank IC 标准差；
- Rank ICIR；
- Rank IC 为正的日期比例；
- t 统计量与 p 值；
- 滚动 60 日 Rank IC；
- 分年度表现；
- 不同市场状态表现。

### 8.2 分组收益单调性

`analyze_factor_panel()` 已支持 `group_returns`，但正式实现应按每个交易日单独分组：

```text
当日模型分数
→ 横截面分为 Q1～Q5
→ 计算各组下一期收益
→ 对时间取平均
```

不能将所有日期混合后统一分组。

理想结果：

```text
Q1 < Q2 < Q3 < Q4 < Q5
```

或者稳定的反向单调关系。

### 8.3 换手与交易成本

当前换手图表位于：

```text
src/quant_platform/report_charts.py
```

Phase 9 尚未把完整 A 股成本模型扣入模型组合收益。正式流程应为：

```text
预测分数
→ 目标组合
→ 与上一期持仓比较
→ 计算买卖换手
→ 扣除税费、滑点和冲击成本
```

至少考虑：

- 买卖佣金；
- 最低佣金；
- 卖出印花税；
- 过户费；
- 滑点；
- 市场冲击；
- 100 股整数倍；
- T+1；
- 停牌；
- 涨跌停；
- 成交额参与率。

### 8.4 市场状态分组

当前独立模块：

```text
src/quant_platform/market_state.py
```

应在模型测试期按当时可见数据识别：

- 上涨趋势；
- 下跌趋势；
- 震荡；
- 高波动；
- 低波动；
- 流动性紧张。

分别报告：

| 市场状态 | Rank IC | Top 组收益 | 换手率 | 最大回撤 |
|---|---:|---:|---:|---:|
| 趋势上涨 | 待计算 | 待计算 | 待计算 | 待计算 |
| 震荡 | 待计算 | 待计算 | 待计算 | 待计算 |
| 高波动 | 待计算 | 待计算 | 待计算 | 待计算 |

市场状态本身不得使用未来数据。

## 9. 模型准入门

当前最重要的缺口是：即使样本外 Rank IC 为负，系统仍可能生成候选池、PPO 仓位和纸面订单。

应在以下文件中、模型评估和候选池之间增加准入判断：

```text
src/quant_platform/phase9_service.py
```

建议新增 `ModelGate`，示意规则：

```python
if (
    out_of_sample_rank_ic <= 0
    or rank_icir < minimum_icir
    or not group_returns_monotonic
    or net_return_after_cost <= 0
):
    candidates = []
    sizing_adjustment = 0.0
    targets = []
    orders = []
    status = "model_gate_rejected"
```

阈值不应凭经验一次固定，应使用训练集与验证集确定，并在测试集只执行一次最终检验。

模型被拒绝时，界面应显示：

```text
模型准入：失败
候选池：仅供诊断
PPO：跳过
目标仓位：0
现金：100%
订单：0 笔
允许正式仿真：否
允许实盘：否
```

## 10. 推荐实施顺序

1. **增加模型准入门**  
   防止无效模型继续生成订单，风险优先级最高。

2. **修改单因子 IC 诊断**  
   只使用训练期筛选，测试期仅评价。

3. **接入缺失检查、MAD 去极值和截面标准化**  
   放在 Qlib 因子计算与 LightGBM 之间。

4. **增加相关性聚类**  
   删除 `VSUMP/VSUMN/VSUMD` 等重复信息。

5. **补充行业和市值数据**  
   A 股个股执行中性化；ETF 使用分类暴露控制。

6. **完善模型组合回测**  
   加入每日横截面分组、真实成本和成交限制。

7. **接入市场状态分组**  
   形成模型适用环境与失效环境报告。

## 11. 模块职责总结

```text
数据中心
├── 原始数据质量
├── 缺失、重复、时间戳
├── 复权与历史股票池
└── 可交易性过滤

因子处理流水线
├── 因子缺失检查
├── 去极值
├── 中性化
└── 截面标准化

因子研究模块
├── IC / Rank IC / ICIR
├── 分组收益
├── 因子衰减
├── 相关性聚类
└── 因子筛选

LightGBM / DoubleEnsemble
├── 训练
├── 验证集早停
├── 特征重要性
└── 输出预测分数

模型评估与回测
├── 样本外 Rank IC
├── 分组收益单调性
├── 换手和交易成本
├── 市场状态表现
└── 最大回撤

风险层
├── 模型准入
├── 仓位上限
├── 集中度限制
├── 流动性限制
└── 未达标保持现金
```

## 12. 最终原则

正确流程不是：

```text
计算所有因子 → LightGBM → 选出前三名 → 下单
```

而是：

```text
训练期内清洗和筛选因子
→ LightGBM / DoubleEnsemble 学习
→ 完全未参与训练的时间段验证
→ 检查分组收益、市场状态和交易成本
→ 通过模型准入门
→ 才允许进入候选池和纸面交易
```

若样本外 Rank IC 为负、分组收益不单调，或扣除成本后无收益，即使模型每天都能输出排名前三，也应保持现金并停止生成订单。

---

回测收益不代表未来收益。本系统只用于可重复、可验证、风险可控的量化研究与交易辅助，不能承诺盈利。
