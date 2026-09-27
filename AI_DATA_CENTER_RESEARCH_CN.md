> **R4 实验与数据复核（2026-09-27）**：在合成 730 日驱动和初始库容条件下，HA 的日/年/小时代理边界为 1150/1300/1100 MW，CO 为 1450/1700/1350 MW。HA 去除再生水目标后的物理年边界为 1650 MW；HA 的 1300 MW 年边界是物理约束与政策目标的联合口径。HA @1000 MW 两年末水库较初始少 27,527 ML；CO @1000 MW 少 1,604 ML。10 年、365 日合成年份序列的“末库容 ≥ 初始值 90% 且居民服务可靠度 ≥99%”联合边界为 HA 250 MW、CO 900 MW；这是一条单一合成水文诊断。按设施电量计算的全期平均日均接网占用为 HA 0.591、CO 0.584，逐日峰值字段为 HA 0.694、CO 0.685。合成驱动已加入 `city_non_ai_electricity_mw` 城市基线负荷列；真实城市节点容量、多序列水文和现场率定仍待完成。当前研究主线为瓶颈迁移机制，1024 样本最终概率估计和 16 个完整网格抽样点复核均已完成。
>
> 详细复核见 [`Nature级研究审查与修正计划_R3.md`](../01_方案与设计文档/Nature级研究审查与修正计划_R3.md)。
# AI-UWM：AI 算力—城市水资源承载力研究指南

## 1. 研究定位与公开依据

本框架保持战略级、日尺度、质量守恒的建模定位。公开文献将该类城市水系统性能模型定义为用于城市综合水系统长期规划和代谢绩效评估的工具，因此本项目把 AI 数据中心作为城市需求、排水和回用闭环中的一等组件，而不是孤立的 WUE 计算器：

- [Behzadian 等（2014）：城市水系统定量绩效模型公开论文，DOI 10.5194/dwes-7-63-2014](https://doi.org/10.5194/dwes-7-63-2014)
- [城市水代谢相关论文记录（UWL 机构库 1824）](https://repository.uwl.ac.uk/id/eprint/1824/)
- [再生水回用与城市水系统综合框架研究（PMC7028841）](https://pmc.ncbi.nlm.nih.gov/articles/PMC7028841/)

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
| `ai_capacity.py` | 0–2 GW连续扫描、JSON约束判定、最大安全AI容量、逐约束边界、瓶颈迁移与一阶干预扫描 |
| `ai_scenarios.py` | 容量×冷却×水源×水文气候×基础设施矩阵、hot+drought、离散Pareto策略 |
| `sensitivity.py` | Morris、Sobol、Monte Carlo概率承载边界和指定容量超限概率 |
| `cawcc.py` | 开题报告 S0–S3/G0–G3 场景标签、Table-1约束生成、训练/推理日内负荷与小时峰值代理 |
| `research.py` | S/G状态压力矩阵、AI/工业对照、小时边界代理、无量纲裕度、稳健性矩阵与参数证据链 |
| CLI/API/Toolkit | 批量运行、自动化、Studio调用及结果导出 |
| Studio | AI节点参数编辑、AI Water KPI及日序列展示 |

## 3. 核心核算

每日 IT 电量：`E_IT = InstalledCapacity × LoadFactor × 24`。园区电量：`E_facility = PUE × E_IT`。湿式散热量经汽化潜热换算为蒸发量，随后：

```text
GrossMakeup = Evaporation + Drift + Blowdown
ExternalMakeup = Consumption + ReturnFlow
Withdrawal + StorageStart = Consumption + ReturnFlow + StorageEnd
```

现场 WUE 定义为 `ExternalWithdrawal / IT energy`，过程补水 WUE 定义为 `ExternalMakeup / IT energy`；两者均由过程结果反算，绝不再次驱动补水，因此不会重复计算。干冷与 liquid-to-air 的现场日常补水为零；liquid-to-water仍需由最终散热侧配置决定湿式比例，不能简单等同零耗水。

当 `coc_mode=quality_limited`：

```text
CoCmax = min(QualityLimit_i / BlendedSourceConcentration_i)
CoC = min(DesignCoC, CoCmax)
```

缺少水质数据时会标记 `quality_coc_fallback=true` 并回退设计CoC。

## 4. 直接运行示例

```powershell
python -m aiuwm.cli validate examples\ai_data_center\project.json
python -m aiuwm.cli run examples\ai_data_center\project.json --output output\ai_city
python -m aiuwm.cli ai-scan examples\ai_data_center\project.json --min-mw 0 --max-mw 2000 --step-mw 100 --output output\ai_capacity_scan.csv
python -m aiuwm.cli ai-threshold examples\ai_data_center\project.json --spec examples\ai_data_center\capacity_constraints.json --output output\ai_threshold
python -m aiuwm.cli ai-bottlenecks examples\ai_data_center\project.json --spec examples\ai_data_center\capacity_constraints.json --output output\ai_bottlenecks
python -m aiuwm.cli ai-intraday examples\ai_data_center\project.json --output output\ai_intraday
python -m aiuwm.cli ai-states examples\ai_data_center\project.json --output output\ai_states
python -m aiuwm.cli ai-industrial examples\ai_data_center\project.json --output output\ai_industrial
python -m aiuwm.cli ai-provenance examples\ai_data_center\project.json --output output\ai_provenance
```

主要产物：

- `data_center_daily/weekly/monthly/annual.csv`
- `ai_capacity_scan.csv`
- `ai_capacity_threshold.csv/json`
- `pollutant_daily.csv` 中的 AI blowdown 和下游 WWTW 负荷
- `constraint_boundaries.csv`：WTW、WWTW、回用、供水可靠性以及可选本地接入/电网接入的首个失效点
- `interventions.csv`：单因素扩容、回用或接入能力干预后的承载边际
- `ai_hourly_proxy.csv`：由日尺度结果按训练/推理 profile 展开的守恒小时代理；用于极端峰值敏感性，不替代小时遥测或水力模型
- `state_pressure_matrix.csv`：S0–S3 × G0–G3 条件化响应矩阵
- `industrial_control.csv`：AI 与等效普通工业控制对照
- `parameter_provenance.csv`：参数来源、单位、年份、证据类型和不确定性

Python接口：

```python
from aiuwm import (
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

## 9. R3 统计与承载判定口径

`find_ai_carrying_capacity()` 在首个失败容量点返回全部越界约束，并按无量纲违约程度报告主导约束，避免 JSON 字典顺序造成瓶颈伪影。概率实验使用 250 MW 锚点并在约束转折处恢复 50 MW 网格；物理 CAWCC 与 `reclaimed_water_substitution_ratio` 政策阈值分开估计。新增末库容诊断后，设施吞吐边界与供水可持续性边界分别报告：在两年合成驱动、初始库容 65,000 ML 的条件下，期末库容不低于初始值的最后通过容量为 D-HA 150 MW、D-CO 900 MW；在单一 10 年、365 日合成年份序列下，末库容保留 90% 且居民服务可靠度 ≥99% 的联合边界为 250/900 MW。Morris `mu*` 按每个输入的取值区间归一化后才允许跨参数排名。旧 `probabilistic_large_*` 只用于审计历史；当前 `probabilistic_corrected_*` 已完成每域 1024 样本最终估计。
