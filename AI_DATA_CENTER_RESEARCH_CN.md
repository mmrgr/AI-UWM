# WaterMet² AI算力—城市水资源承载力研究指南

## 1. 研究定位与公开依据

本扩展保留 WaterMet² 的战略级、日尺度、质量守恒建模定位。WaterMet² 原始开放论文将模型定义为用于城市综合水系统长期规划和代谢绩效评估的工具，因此本项目把 AI 数据中心作为城市需求、排水和回用闭环中的一等组件，而不是孤立的 WUE 计算器：

- [WaterMet² 原始开放论文（DWES, 2014）](https://dwes.copernicus.org/articles/7/63/2014/)
- [WaterMet² 城市水代谢论文记录](https://repository.uwl.ac.uk/id/eprint/1824/)
- [Water reuse 与 WaterMet² 综合框架研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC7028841/)

数据中心侧采用“IT容量→负荷→电量→热负荷→冷却→补水”的显式链。美国能源部资料强调冷却塔补水、排污和浓缩倍数对连续冷却负荷的重要性；ASHRAE资料指出再生水水质会影响结垢风险和排污强度。本实现据此提供固定/水质约束 CoC、再生水与饮用水组合及 blowdown 入污水厂，但不虚构复杂化学反应：

- [DOE：数据中心冷却水效率](https://www.energy.gov/cmei/femp/cooling-water-efficiency-opportunities-federal-data-centers)
- [ASHRAE：Cooling-water treatment](https://handbook.ashrae.org/Handbooks/A23/SI/A23_Ch50/a23_ch50_si.aspx)
- [LBNL：2024 U.S. Data Center Energy Usage Report](https://eta-publications.lbl.gov/publications/2024-lbnl-data-center-energy-usage-report)

LBNL同时区分现场冷却耗水与发电相关的间接水足迹。本项目严格把 `offsite_electricity_water_ml` 放在扩展足迹账户中，不加入城市现场物理水量平衡。

## 2. 软件组成

| 模块 | 功能 |
|---|---|
| `data_center.py` | 容量计划、fixed/timeseries/profile负荷、PUE、湿球温度、六类冷却、冷却塔水量、水质约束CoC、储水与直接/间接水账户 |
| `data/ai_data_center_database.json` | 带来源、年份、单位与不确定性分布的PUE/WUE/CoC/漂水/水质/电网水强度参数先验 |
| `full_engine.py` | 将AI补水接入真实 potable/reuse 容量与优先级，将blowdown及污染物送入 sewer→WWTW→次日reuse闭环 |
| `ai_metrics.py` | Withdrawal/Consumption/Return Flow、FDR/RWS/Circularity、最大日/P95/7日/夏季峰值、设施利用率、Baseline差分 |
| `ai_capacity.py` | 0–2 GW连续扫描、JSON约束判定、最大安全AI容量与限制约束 |
| `ai_scenarios.py` | 容量×冷却×水源×水文气候×基础设施矩阵、hot+drought、离散Pareto策略 |
| `sensitivity.py` | Morris、Sobol、Monte Carlo概率承载边界和指定容量超限概率 |
| CLI/API/Toolkit | 批量运行、自动化、Studio调用及结果导出 |
| Studio | AI节点参数编辑、AI Water KPI及日序列展示 |

## 3. 核心核算

每日 IT 电量：`E_IT = InstalledCapacity × LoadFactor × 24`。园区电量：`E_facility = PUE × E_IT`。湿式散热量经汽化潜热换算为蒸发量，随后：

```text
GrossMakeup = Evaporation + Drift + Blowdown
ExternalMakeup = Consumption + ReturnFlow
Withdrawal + StorageStart = Consumption + ReturnFlow + StorageEnd
```

`WUE = ExternalMakeup / IT energy` 仅由过程结果反算，绝不再次驱动补水，因此不会重复计算。干冷与 liquid-to-air 的现场日常补水为零；liquid-to-water仍需由最终散热侧配置决定湿式比例，不能简单等同零耗水。

当 `coc_mode=quality_limited`：

```text
CoCmax = min(QualityLimit_i / BlendedSourceConcentration_i)
CoC = min(DesignCoC, CoCmax)
```

缺少水质数据时会标记 `quality_coc_fallback=true` 并回退设计CoC。

## 4. 直接运行示例

```powershell
python -m watermet2_repro.cli validate examples\ai_data_center\project.json
python -m watermet2_repro.cli run examples\ai_data_center\project.json --output output\ai_city
python -m watermet2_repro.cli ai-scan examples\ai_data_center\project.json --min-mw 0 --max-mw 2000 --step-mw 100 --output output\ai_capacity_scan.csv
python -m watermet2_repro.cli ai-threshold examples\ai_data_center\project.json --spec examples\ai_data_center\capacity_constraints.json --output output\ai_threshold
```

主要产物：

- `data_center_daily/weekly/monthly/annual.csv`
- `ai_capacity_scan.csv`
- `ai_capacity_threshold.csv/json`
- `pollutant_daily.csv` 中的 AI blowdown 和下游 WWTW 负荷

Python接口：

```python
from watermet2_repro import (
    compare_baseline_ai, find_ai_carrying_capacity,
    generate_ai_scenario_matrix, morris_sensitivity,
    pareto_ai_strategies, probabilistic_ai_capacity_threshold,
    scan_ai_capacity, sobol_sensitivity, summarize_ai_water_kpis,
)
```

## 5. Baseline 与反事实

Baseline只移除 `kind=data_center` 的组件，人口、普通工业、气候和基础设施时间变化保持一致。`compare_baseline_ai()`逐指标输出 `baseline`、`ai_scenario` 和 `delta`，从而避免把城市自然增长误判为AI影响。

## 6. 五维情景与承载边界

标准水平为：

- 容量：100、300、500、800、1000、1500、2000 MW；
- 冷却：evaporative、efficient_evaporative、hybrid、dry、liquid_to_air、liquid_to_water；
- 水源：100% potable、70/30、50/50、20/80、reclaimed-first；
- 水文气候：normal、hot、drought、hot+drought；
- 基础设施：current、reuse expansion、WTW expansion、leakage reduction、combined upgrade。

`find_ai_carrying_capacity()`接受任意以 `>=`、`>`、`<=`、`<`、`==` 表示的KPI约束，返回最大安全容量、首个失败容量、限制约束、超限幅度和阈值区间。`probabilistic_ai_capacity_threshold()`对参数路径采样并逐次重算阈值；`capacity_exceedance_probability()`直接回答拟建1 GW等规模的超限概率。

## 7. 可直接回答的论文问题

1. 500 MW、1 GW、2 GW分别新增多少取水、耗水和回流水：容量扫描表。
2. 新增水来自淡水还是再生水：`potable_water_ml`、`reclaimed_water_ml`。
3. 城市代谢结构是否变化：Baseline/AI 的FDR、RWS、Circularity差分。
4. AI冷却峰值是否与城市峰值重合：`coincident_city_ai_peak`及日序列。
5. hot+drought何时出现压力：复合情景日KPI与设施利用率。
6. 最大安全MW/GW：约束阈值结果。
7. 低水耗冷却增加多少安全容量：按冷却技术分别求阈值。
8. 提高再生水比例增加多少容量：水源情景阈值差。
9. 扩建再生水还是传统供水更有效：基础设施情景差分与Pareto前沿。
10. 哪些参数主导风险：Morris `mu_star`、Sobol `S1/ST`。
11. 1 GW突破边界概率：Monte Carlo阈值分布与超限概率。
12. 最优“容量—冷却—水源—设施”组合：`pareto_ai_strategies()`。

## 8. 模型边界

这是战略水代谢模型，不包含EPANET压力、水力瞬变、二维洪水或CFD。水质模块用于约束CoC和追踪战略污染负荷，不替代冷却塔详细化学设计。示例参数用于可重复验证，具体城市研究必须换成当地设施容量、逐日气象/入流、供水规则、电力水足迹与冷却设备数据，并进行校准和不确定性分析。
