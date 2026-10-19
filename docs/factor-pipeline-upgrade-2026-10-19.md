# 因子—模型流水线升级总结（2026-10-19）

> 项目：`personal-quant-platform`
> 范围：Phase 9 组合研究流水线、Qlib Alpha158、LightGBM/DoubleEnsemble、因子处理、模型准入、成本与成交约束
> 依据：`docs/factor-pipeline-integration-guide.md`（第 6/7 步）、`docs/factor-pipeline-remaining-improvements.md`（批次 A/B/C + Section 13）
> 安全原则：全程 paper-only，真实交易关闭，`live_trading_enabled=false`、`ai_execution_tools_enabled=false`、`execution_enabled=false`

---

## 一、本轮完成的工作

### 1. 分组收益单调性 + 交易成本（指南第 6 步）

- `qlib_model.compute_group_returns`：按**每日横截面**把模型分数切成 Q1~Q5，计算各组下一期收益后对时间取平均，输出单调性判定、多空价差（Q5−Q1）。
- `report_charts.compute_report_charts`：扣除换手成本，新增扣费后净值 `strategy_nav_net` 与 `net_return_after_cost`/`cost_drag`。

### 2. 市场状态分组（指南第 7 步 → 剩余事项批次 B 多维化）

- `report_charts.compute_market_state_report`：以等权 `adj_close` 为基准，用滚动窗口（无未来数据）识别市场状态。
- 由单一 6 类升级为**多维标签**：趋势（涨/跌/震荡）× 波动（高/中/低）× 流动性（正常/紧张）× 异常，报告每状态的 Rank IC、Top 组收益、换手率、最大回撤。

### 3. 扣费后净收益纳入模型准入

- `_model_gate` 增加 `net_return_after_cost <= 0` 拦截，理由结构化返回。

### 4. 稳健性：多种子平均 + Walk-forward 滚动训练

- `qlib_alpha158.run_alpha158` 新增 `n_seeds`/`seed`：多种子训练取平均分数，`seed_scores` 保留各种子分数；`run_phase9_alpha158` 计算跨种子 Rank IC 稳定性（均值±标准差）。
- 新增 `walk_forward`/`n_walk_folds`：扩展窗口滚动训练，拼接完整样本外分数序列，每折只用过去数据，无未来泄漏。

### 5. 因子筛选前移到训练之前（剩余事项 批次 A · P0）

- 新增 `factor_selection.select_factors`：在**训练期**数据上做 IC 诊断 + 相关性聚类去冗余，输出 `raw_count → selected_count` 与逐因子 keep/drop/冗余 标签。
- `run_alpha158` 改为 `原始因子 → 预处理 → 训练期筛选 → 仅保留因子训练`；baseline 做筛选、ensemble 复用同一 `selected` 清单。
- 删除旧 `compute_factor_ic_report`，由 `select_factors` 单一来源取代。

### 6. 时间切分 purge/embargo（批次 A · P0）

- `_build_dataset` 改为显式六边界；`_segment_boundaries` 在 train/valid/test 之间按交易日插入 `purge=1`（剔除跨界标签样本）+ `embargo=1`（隔离区），防止 1 日前向标签跨界泄漏。

### 7. 多维模型准入门（批次 A · P0）

- `_model_gate` 升级为六维结构化判定，返回 `rules` 明细（前端逐条 ✓/✗）：
  1. 样本外 Rank IC（> `min_ic`，默认 0.02）
  2. Rank ICIR（≥ `min_icir`，默认 0.15）
  3. Rank IC 正日比例（≥ 0.5，`evaluate_scores` 新增 `positive_ratio`）
  4. 分组收益单调性（`compute_group_returns` 新增 `monotonicity_score`/`top_relative`）
  5. 扣费后净收益（> 0）
  6. 最大回撤（≥ 阈值）

### 8. 拆分交易成本（批次 B · P1）

- 新增 `cost_model.build_cost_model`：按资产类别拆分买入/卖出佣金、印花税（ETF 免）、过户费、滑点、最低佣金，给出 `round_trip_rate`（A 股约 0.302%、ETF 约 0.12%）。
- `compute_report_charts` 输出 `cost_breakdown`（累计佣金/印花税/过户费/滑点）与 `cost_label`。

### 9. 行业时点口径 + ETF 暴露（批次 B · P1）

- `_build_neutralization_data` 返回第三元 `meta`：`industry_source`（静态最新）、`industry_effective_date`、`market_cap_date`、免责声明；结果新增 `neutralization_meta`。
- 新增 `_ETF_CATEGORIES` + `_classify_etf_exposure`：标注候选 ETF 类别/跟踪指数，输出 `overlap_categories`（诊断，不改变候选）。

### 10. 统计显著性 + 版本指纹（批次 C · P2）

- 图表指标新增 `ic_t_stat`、`ic_p_value`、`ic_positive_ratio`、`rolling_ic_mean`（20 日）。
- 运行历史新增 `fingerprint`：git 版本、数据源、起止日期、因子原始/保留数、min_ic/min_icir、seed/n_seeds/walk_forward、purge_embargo、成本模型、标签定义。

### 11. 成交限制与容量（Section 13 · P2）

- 新增 `capacity.estimate_capacity`：按资产输出日均成交额、容量、**预期成交天数**、**平方根冲击成本**，并给出最弱流动性标的与总容量；接入 `analyze_factor_panel` 的 `capacity` 输出。
- 新增 `execution_constraints`：`apply_lot_size`（100 股整数倍）、`is_limit_locked`（涨停不可买/跌停不可卖）、`participation_notional_cap`、`t_plus_1_sellable`。
- `PaperMatchingEngine` 新增 `lot_size` 并接入成交取整；`phase9_payload` 对股票/ETF 建仓设 `lot_size=100`。

---

## 二、新增 / 修改文件

**新增**
- `src/quant_platform/factor_selection.py`
- `src/quant_platform/cost_model.py`
- `src/quant_platform/capacity.py`
- `src/quant_platform/execution_constraints.py`
- `tests/test_capacity.py`、`tests/test_cost_model.py`、`tests/test_execution_constraints.py`

**修改**
- `src/quant_platform/qlib_alpha158.py`、`qlib_model.py`、`phase9_service.py`、`report_charts.py`
- `src/quant_platform/factor_analysis.py`、`http_api.py`、`execution.py`、`simulation.py`
- `tests/test_qlib_alpha158.py`、`test_qlib_model.py`、`test_phase9_service.py`、`test_report_charts.py`、`test_execution.py`
- `web/index.html`

---

## 三、测试

- 全量 **441 项单元测试通过**（`arch -arm64 .venv-qlib/bin/python -m unittest discover -s tests`）。
- 新增覆盖：因子筛选、purge/embargo 边界、多维准入规则、成本模型、容量、成交限制原语、整手成交。

---

## 四、对跑测的影响

- **更可信**：因子筛选只读训练期、purge/embargo 防泄漏、Walk-forward 可选，样本外 Rank IC 更接近真实。
- **更严格**：准入从 3 条升到 6 条，微弱正 Rank IC 不再下单，更可能「保持现金、0 订单」；网页逐条展示拦截原因。
- **更慢**：因子筛选 + 默认 3 种子约使训练时长 ×3；勾选滚动训练再翻几倍。
- **更可读/可复现**：单调性、多维市场状态、成本明细、统计显著性、版本指纹全部落盘，复盘可还原完整条件。
