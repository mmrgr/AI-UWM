# WaterMet² 开放复现与城市水系统分析框架

这是依据 Behzadian 与 Kapelan（2015）论文、TRUST D33.2 官方手册和 Oslo 案例报告重写的 Python 实现。它覆盖公开文献描述的 WaterMet² 分析链：日尺度水量与污染物质量守恒、四级空间表达、供水—用水—雨洪—污水—回用、能源/材料/化学品、环境影响、成本、资源回收、干预方案、校准、不确定性、CP/AHP 多准则排序和场景决策支持。

原始 WaterMet² 是闭源 C# 软件，论文使用的 Oslo SCADA 日序列也未公开，因此本项目不能声称与原二进制逐行或逐日数值完全相同；但公开方法和功能都已有可审计实现。完整对照见 [FUNCTIONAL_COVERAGE_CN.md](FUNCTIONAL_COVERAGE_CN.md)，论文案例复现边界见 [REPRODUCTION_REPORT_CN.md](REPRODUCTION_REPORT_CN.md)。

## Windows 一键打开

日常使用只需双击项目根目录中的：

```text
启动WaterMet2 Studio.cmd
```

程序会自动检查依赖、构建网页、启动模型服务并打开
<http://127.0.0.1:8000>。现在前端和 API 由同一个 Python 进程提供，不再需要手动启动两个终端。

第一次启动可能需要几分钟安装依赖；以后通常会直接打开。关闭启动程序窗口即可停止系统。目录用途见 [项目文件说明.md](项目文件说明.md)，自有数据操作教程见 [frontend/README.md](frontend/README.md)。

主要公开依据：

- 2015 论文 DOI：<https://doi.org/10.1016/j.resconrec.2015.03.015>
- 2014 开放论文与模型概述：<https://doi.org/10.5194/dwes-7-63-2014>
- Exeter 官方仓储的功能需求：<https://ore.exeter.ac.uk/repository/handle/10871/17302>
- `references/` 仅发布来源元数据；论文和报告 PDF 因版权原因不随公开仓库上传，可从上述官方地址下载。

## 1. 安装与验证

在本目录打开 PowerShell：

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

当前验证结果为 `49 passed`。安装后可使用 `watermet2` 命令；若环境没有刷新命令入口，可用 `python -m watermet2_repro.cli` 替代。

### 可视化前端

项目包含 React + TypeScript 的 WaterMet² Studio，可编辑系统拓扑、导入 CSV、执行前后端校验、运行同一套 Python 模型，并查看 KPI、桑基图、时间序列、风险和情景对比。

```powershell
python -m pip install -e ".[studio,test]"
watermet2-api
```

另开一个 PowerShell：

```powershell
Set-Location frontend
npm install
npm run dev
```

访问 <http://127.0.0.1:5173>。分析自有数据的完整中文步骤见 [frontend/README.md](frontend/README.md)。

### 一键生成模拟数据并验证整个流程

```powershell
$env:PYTHONPATH='src'
python scripts\run_full_validation.py
```

该脚本不读取预制观测值，而是用固定随机种子从代码生成 731 天天气和水源序列，再执行完整模拟、质量守恒审计、资产失效、全寿命成本、地下水补给、洪涝、23 类风险、已知真值校准、情景分析、Monte Carlo、CP/AHP、完整 DSS 和 Pareto 优化。当前为 `29/29` 项通过。结果写入 `validation_artifacts/`，中文结论见 `validation_artifacts/VALIDATION_REPORT_CN.md`，机器可读判据见 `validation_artifacts/validation_report.json`。

## 2. 先运行完整示例

```powershell
watermet2 validate examples\demo_full\project.json
watermet2 run examples\demo_full\project.json --output output\demo
```

完整示例包含两套水源—WTW—水池—配水链、室内/工业/季节需求、RWH、GWR、中心回用、混合排水、两座 WWTW、受纳水体、BOD/TSS/TN/TP、污泥与资源回收、管道更新和分期干预。

### AI算力与城市水承载力示例

项目现已支持一等 `data_center` 组件、六类冷却技术、potable/reclaimed联合供水、冷却塔质量守恒、blowdown→WWTW→reuse闭环、峰值KPI、0–2 GW扫描、承载边界、复合情景、Monte Carlo、Morris/Sobol及Pareto策略。完整方法见 [AI_DATA_CENTER_RESEARCH_CN.md](AI_DATA_CENTER_RESEARCH_CN.md)。
研究参数先验位于 `data/ai_data_center_database.json`；示例通过 `ai_database_file` 引用，项目内显式参数始终优先。

```powershell
watermet2 run examples\ai_data_center\project.json --output output\ai_city
watermet2 ai-scan examples\ai_data_center\project.json --min-mw 0 --max-mw 2000 --step-mw 100
watermet2 ai-threshold examples\ai_data_center\project.json --spec examples\ai_data_center\capacity_constraints.json
watermet2 ai-sensitivity examples\ai_data_center\project.json --spec examples\ai_data_center\sensitivity.json
watermet2 ai-probabilistic-threshold examples\ai_data_center\project.json --spec examples\ai_data_center\probabilistic_threshold.json
watermet2 ai-exceedance --samples output\ai_probabilistic_threshold.csv --proposed-mw 1000
watermet2 ai-bottlenecks examples\ai_data_center\project.json --spec examples\ai_data_center\capacity_constraints.json
watermet2 ai-intraday examples\ai_data_center\project.json --output output\ai_intraday
watermet2 ai-states examples\ai_data_center\project.json --output output\ai_states
watermet2 ai-industrial examples\ai_data_center\project.json --output output\ai_industrial
watermet2 ai-hourly-boundary examples\ai_data_center\project.json --spec examples\ai_data_center\hourly_boundary.json --output output\ai_hourly_boundary
watermet2 ai-robustness examples\ai_data_center\project.json --spec examples\ai_data_center\robustness.json --output output\ai_robustness
watermet2 ai-provenance examples\ai_data_center\project.json --output output\ai_provenance
```

Studio API 还提供 `/api/ai/baseline`、`/api/ai/sensitivity`、
`/api/ai/probabilistic-threshold`、`/api/ai/exceedance-probability`、
`/api/ai/bottlenecks`、`/api/ai/intraday-proxy`、`/api/ai/hourly-boundary`、
`/api/ai/states`、`/api/ai/industrial-control`、`/api/ai/robustness` 和
`/api/ai/provenance`，用于把基准—反事实、参数敏感性、概率承载边界、
瓶颈迁移/干预边际、S/G状态、工业对照、稳健性和证据链直接接入论文实验流程。
`ai-intraday` 使用可替换的训练/推理日内负荷假设，并严格保持每日水量守恒；获得遥测后可直接替换 profile。

若只想重现论文 Oslo 图表：

```powershell
python scripts\run_reproduction.py
```

## 3. 用自己的数据：推荐流程

### 3.1 复制模板

复制以下三个文件到新目录：

- `examples/demo_full/project.json`：系统结构、参数和方案；
- `examples/demo_full/timeseries.csv`：每日外部驱动数据；
- `data/watermet2_database.json`：官方附录的能源、材料、化学品和修复系数。

在新 `project.json` 中保留：

```json
{
  "database_file": "../../data/watermet2_database.json",
  "timeseries_file": "timeseries.csv",
  "simulation": {"start": "2020-01-01", "end": "2024-12-31"}
}
```

路径相对于 `project.json` 所在目录。

### 3.2 准备日序列 CSV

第一列必须是 `date`，日期不能重复或缺日。其他列只在 JSON 中按名称引用。典型格式：

```csv
date,rainfall_mm,temperature_c,source_inflow_ml,population
2020-01-01,3.2,5.6,210.4,500000
2020-01-02,0.0,4.9,198.1,500010
```

单位约定：水量 `ML/day`，储量 `ML`，降雨/蒸散 `mm/day`，温度 `°C`，污染物质量 `kg/day`，能量 `kWh`，货币默认为 EUR。`1 ML = 1000 m³`。

水源用 `inflow_column` 关联入流列；Local area 用 `weather_columns` 关联雨量和温度；人口既可用 `population_column`，也可使用 `base_population`，或 `number_of_properties × occupancy_people_property`。

### 3.3 定义组件与连接

在 `components` 中定义所有设施。支持的 `kind`：

| kind | 含义 | 常用字段 |
|---|---|---|
| `water_resource` | 水库、河流、地下水、海水或外部进口水 | `resource_type`, `capacity_ml`, `initial_ml`, `inflow_column`, `abstraction_capacity_ml_day` |
| `supply_conduit` | 原水输水 | `daily_capacity_ml`, `leakage_fraction` |
| `wtw` | 给水厂 | `daily_capacity_ml`, `loss_fraction` |
| `trunk_main` | 清水干管 | `daily_capacity_ml`, `leakage_fraction` |
| `service_reservoir` | 服务水池 | `capacity_ml`, `initial_ml`, `loss_fraction` |
| `distribution_main` | 配水系统 | `daily_capacity_ml`, `leakage_fraction` |
| `reuse` | RWH、GWR 或中心再生水 | `reuse_type`, `scale`, `capacity_ml`, `eligible_demands` |
| `sewer` | sanitary、storm 或 combined 排水 | `sewer_type`, `capacity_mode`, `capacity_ml`, `release_a`, `release_b` |
| `wwtw` | 污水厂 | `daily_capacity_ml`, `pollutant_removal_fraction` |
| `receiving_water` | 河流、湖泊或海域 | 无必填水力参数 |
| `data_center` | AI算力中心 | `installed_it_capacity_mw`, `load_factor`, `pue_mode`, `cooling`, `water_sources` |

每条 `supply_paths` 必须从 `water_resource` 开始，以 `distribution_main` 结束。同一 Local area 的所有路径 `allocation` 合计必须为 1。污水设施通过 `wastewater_connections` 连接，分流比例用 `fraction`。

也可使用更接近官方 Toolkit 的逐边拓扑 `supply_connections`。引擎会自动编译所有水源到 Local area 的路径：

```json
[
  {"from":"WR1","to":"WTW1","allocation":1.0},
  {"from":"WTW1","to":"SR1","allocation":1.0},
  {"from":"SR1","to":"DM1","allocation":1.0},
  {"from":"DM1","to":"LA1","allocation":1.0}
]
```

每个节点所有入边的 `allocation` 必须合计为 1。

### 3.4 定义区域、需求和污染物

每个 `local_areas` 可归属一个 `subcatchment`。`demand_profiles` 可逐项表示淋浴、厕所、工业、灌溉等：

```json
{
  "name": "toilet",
  "base_value": 45,
  "unit": "l_capita_day",
  "annual_growth": 0.01,
  "return_fraction": 0.95
}
```

支持 `l_capita_day`、`m3_day`、`ml_day`；也可用 `column` 直接读取实测日需求。季节需求使用 `start_mmdd`、`end_mmdd`，温度敏感需求使用 `temperature_sensitivity`。

地表径流在 `surfaces` 中按面积比例、径流系数和 EMC 定义。生活污水污染负荷用 `sanitary_pollutant_load_kg_capita_day`。污染物名称不限于 BOD/TSS/TN/TP；只要各输入、去除率和影响系数使用同一名称，就能追踪任意污染物。

地表可增加 `recharge_fraction`，模型会把未形成径流的有效降水按该比例计入 `aquifer_recharge_ml`。排水组件可用 `flood_depth_m`、`flood_shape_factor`、`street_flow_width_m` 和 `dangerous_velocity_m_s`，把溢流体积按简化洪涝锥换算为淹没面积及高流速暴露面积。

完整四级层级可显式定义：

```json
{
  "subcatchments": {"SUB1": {"local_areas":["LA1"]}},
  "indoor_areas": {
    "HOUSE_TYPE_A": {
      "local_area":"LA1",
      "number_of_properties":1000,
      "occupancy_people_property":2.5,
      "demand_profiles":[
        {"name":"shower","base_value":45,"unit":"l_capita_day"}
      ]
    }
  }
}
```

对应结果为 `subcatchment_daily.csv` 和 `indoor_daily.csv`。官方气象模式支持 `precipitation_type`、`snow_depth`、`wind_speed`、`sunshine_hours`、`relative_humidity` 和可选 `vapour_pressure`，通过 `weather_columns` 映射。具备完整字段时采用日尺度 Penman–Monteith 蒸发，否则回退到简化温度法。RWH 的 `source_surfaces` 可限定屋面、道路等收集面。

### 3.5 配置资源消耗、影响与回收

任一活动组件可配置：

- `electricity_kwh_m3`、`fossil_fuels`：运行能耗；
- `energy_generation_kwh_m3`：水轮机等发电及替代收益；
- `chemical_doses_kg_m3`：药剂投加；
- `variable_cost_eur_m3`、`fixed_cost_eur_year`：运行成本；
- `direct_emissions`：CH4、N2O、CO2、NH3、NO2、SO2 及污泥逸散；
- `resource_recovery`：沼气、电、热、硝酸铵、过磷酸钙、尿素或自定义副产品。

管材存量写入 `pipelines`，更新工程写入 `pipeline_events`。支持开挖、PE/PP 内衬等方法，自动计算材料、柴油、资本成本和环境影响；`leakage_growth_per_year` 可描述老化导致的年度漏损增长。

给水厂污泥可用 `raw_water_tss_mg_l`、`tss_removal_fraction` 和 `chemical_sludge_yield_kg_per_kg` 配置。WWTW 的 `process_recovery` 可按污泥产气率、沼气热值、CHP效率和出水温差计算沼气、电与热。

WWTW 的 `sludge_process` 可配置 `digestion_mass_reduction_fraction`、`dewatered_dry_solids_fraction`、`dried_dry_solids_fraction` 和 `end_use_fraction`，输出消化干固体、脱水污泥、干化污泥及最终利用生物固体。

管道支持多个 `cohorts` 管龄组、`annual_rehabilitation_rate` 或 `annual_rehabilitation_length_m`；年度更新按 oldest-first 选择，不会因局部修复重置整条管道年龄。

### 3.6 配置资产失效、生命周期成本和设备

设施或管道可配置恒定、线性、指数或 Weibull 失效模型。例如：

```json
{
  "capital_cost_eur": 25000000,
  "investment_date": "2020-01-01",
  "lifetime_years": 50,
  "maintenance_fraction_capital_year": 0.01,
  "failure_model": {
    "model": "exponential",
    "intercept": -2.5,
    "age_coefficient": 0.035,
    "diameter_coefficient": -0.001,
    "basis": "per_km",
    "repair_cost_eur_failure": 4000,
    "duration_hours_failure": 6,
    "material_factors": {"PVC": 0.8, "grey_cast_iron": 1.4}
  }
}
```

组件成本支持投资日、寿命到期重置、通胀、资本回收因子年化、固定维护、资本比例维护和失效维修。`demand_profiles` 还可配置 `electricity_kwh_m3`、`capital_cost_eur`、`lifetime_years`、`investment_date`、`maintenance_cost_eur_year`，用于热水、洗衣等用水设备。

水源 `resource_type` 可设为 `surface`、`groundwater`、`desalination` 或 `imported`；任一供水组件的 `export_fraction` 会进入进出口水量账本。

供水抽象质量指标可用水源 `raw_water_quality_index`、WTW `treated_water_quality_index`、水池 `low_storage_quality_penalty`、配水系统 `quality_decay_per_year` 和资产管龄配置。输出 `tap_water_quality_index`（0–1）及 `excellent/good/acceptable/poor` 等级，并直接进入 R05HZ01。

### 3.7 校验并运行

```powershell
watermet2 validate path\to\project.json
watermet2 run path\to\project.json --output output\my_case
```

先校验能发现缺列、缺日、无效组件、路径断裂和分配比例错误。运行结果包括：

| 文件 | 内容 |
|---|---|
| `system_daily.csv`, `system_annual.csv` | 全系统供水可靠性、漏损、能耗、影响与折现成本 |
| `subcatchment_daily/monthly/annual.csv` | Subcatchment 尺度聚合 KPI |
| `area_daily.csv` | 各 Local area 的需求、饮用水、回用水和缺水 |
| `indoor_daily/monthly/annual.csv` | Indoor/property 类型的需求与供水 |
| `component_daily.csv` | 每个设施的水量、储量、能耗、药剂、排放、成本与回收 |
| `pollutant_daily.csv` | 按组件、污染物和出流类型的质量通量 |
| `recovery_daily.csv` | 回收产品、数量和单位 |
| `material_events.csv` | 管网修复长度、材料、柴油和资本成本 |
| `asset_daily/monthly/annual.csv` | 资产期望失效、停运小时、维修、维护与年化资本成本 |
| `flood_daily/monthly/annual.csv` | 溢流体积、假定水深、淹没面积、街道流速与高流速面积 |
| `risk_daily/weekly/monthly/annual.csv` | 官方 23 个风险代码的多尺度指标、阈值、评估状态、超限和风险分数 |
| `risk_summary.csv` | 各风险的超限天数、概率、最大严重度和累计风险分数 |
| `summary.json` | 可靠性、总缺水、净 GHG 和现值成本摘要 |

系统表中的 `water_demand_ml = delivered_total_ml + unmet_demand_ml`。组件表的 `inflow_ml/outflow_ml` 是设施级通量，不应跨组件直接相加为系统用水量，否则同一滴水会在供水链上被重复计数。

## 4. 校准、风险与方案决策

官方 2012 功能需求把部分风险列为 WaterMet² 直接计算，部分列为“间接估算”或建议由外部 DSS 处理。本实现保留这种边界：水量、CSO、旁路、失效、洪涝和补给来自核心物理账本；低压、伤害、生活习惯、低水质等无法由日尺度战略模型直接求解的项目明确标记为 `deterministic_proxy`，可通过 `risk_thresholds` 和 `risk_settings` 配置，不冒充水力或流行病学模型。

风险阈值示例：

```json
{
  "risk_settings": {
    "baseline_recharge_ml_day": 50,
    "renewable_target_kwh_day": 1000,
    "scarcity_cost_eur_ml": 1000,
    "energy_price_stress_eur_kwh": 0.05,
    "injury_exposure_per_m2": 0.000001
  },
  "risk_thresholds": {
    "R03HZ01": {"threshold": 0.1, "normalization_scale": 1, "consequence_weight": 5},
    "R07HZ01": {"threshold": 1000, "normalization_scale": 10000, "consequence_weight": 10}
  }
}
```

`run` 自动输出全部风险结果。Monte Carlo 的样本表还包含每个风险代码的发生概率和累计分数，因此可以对风险做 P05/P50/P95 批量评估。

### 4.1 校准

观测 CSV 至少含 `date` 和观测列；参数网格、目标表/列及筛选条件参照 `examples/demo_full/calibration.json`：

```powershell
watermet2 calibrate examples\demo_full\project.json `
  --observed examples\demo_full\observed.csv `
  --spec examples\demo_full\calibration.json `
  --output output\calibration
```

输出 NSE、RSR、PBIAS、RMSE、全部试验和 `best_project.json`。规范中可增加 `"frequency":"MS"` 或 `"YS"`，以及 `"aggregation":"sum"`、`"mean"` 或 `"last"`，分别做月/年尺度校准；`result_table` 选择系统、Subcatchment 或区域尺度。应使用不同时间段分别校准和验证，避免用同一序列报告模型能力。

### 4.2 情景与干预

复制基准 JSON 为不同方案，或在 `interventions` 中按日期修改任意已有参数：

```json
{"date":"2030-01-01","set":[{"path":"components.WTW1.daily_capacity_ml","value":500}]}
```

分别运行各方案，将年度可靠性、净 GHG、净酸化、净富营养化和折现成本汇总为 `alternatives.csv`。

### 4.3 Monte Carlo 不确定性

参照 `examples/demo_full/uncertainty.json` 定义参数路径和 `uniform`、`normal`、`triangular`、`lognormal` 或 `choice` 分布：

```powershell
watermet2 uncertainty examples\demo_full\project.json `
  --spec examples\demo_full\uncertainty.json `
  --output output\uncertainty
```

输出每次抽样的输入与 KPI，以及 P05/P50/P95。长期模型先用少量样本调试，再增加到满足稳定性要求的样本数。

### 4.4 多准则方案排序

```powershell
watermet2 rank `
  --alternatives examples\demo_full\alternatives.csv `
  --criteria examples\demo_full\criteria.json `
  --p 2 `
  --output output\ranking.csv
```

这是 Oslo 报告使用的 Compromise Programming。`goal` 可设为 `min` 或 `max`，权重应由利益相关方讨论确定，不应由模型替代价值判断。

AHP 与完整 DSS：

```powershell
watermet2 ahp-rank `
  --alternatives examples\demo_full\alternatives.csv `
  --criteria examples\demo_full\criteria.json `
  --pairwise examples\demo_full\ahp_pairwise.json `
  --output output\ahp_ranking.csv

watermet2 dss examples\demo_full\project.json `
  --spec examples\demo_full\dss.json `
  --output output\dss
```

`dss.json` 可定义多个场景、干预策略、定量指标、外部定性指标和利益相关方偏好组，并同时使用 CP 与 AHP。AHP 输出判断矩阵一致性比率 CR；默认拒绝超过一致性阈值的矩阵。Studio 的“高级分析”页也可直接执行同一工作流。

### 4.5 多目标 Pareto 优化

```powershell
watermet2 optimize examples\demo_full\project.json `
  --spec examples\demo_full\optimization.json `
  --output output\pareto_trials.csv
```

`decisions` 定义离散决策值，`objectives` 的方向为 `min` 或 `max`；输出用 `is_pareto` 标记非支配解。

### 4.6 Python Toolkit

```python
from watermet2_repro import WaterMet2Toolkit

toolkit = WaterMet2Toolkit.open("examples/demo_full/project.json")
toolkit.set_input("components.DM1.leakage_fraction", 0.15)
toolkit.run()
dm1 = toolkit.get_result("component_daily", component_id="DM1")
weekly = toolkit.get_result("system_daily", frequency="weekly")
total_leakage = toolkit.get_result_value("component_daily", "leakage_ml", component_id="DM1")
toolkit.write_results("output/toolkit_case")
```

该接口提供打开/保存工程、读取/修改任意输入和时序列、校验、运行、列举组件、日期筛选、日/周/月/年聚合和标量提取。它提供原 Toolkit 的高层用途，但不声称兼容原 `WaterMet2.dll/Toolkit.dll` ABI 或函数签名。

## 5. 常见问题

- **结果异常大**：最常见原因是把 m³ 当作 ML；所有水量内部统一为 ML。
- **校验提示缺日**：补齐完整日历；未知值不要用删除日期解决，应先插补并记录方法。
- **供水分配报错**：检查同一区域 `supply_paths.allocation` 是否严格合计为 1。
- **缺水为零但水源见底**：检查服务水池初始储量、路径能力和模拟起始期的 warm-up 设置。
- **需要管网压力/管径瞬变**：WaterMet² 是战略代谢模型，不是 EPANET/SWMM 的详细水力替代品。可先用水力模型生成日尺度边界序列，再输入本系统评估长期资源与环境表现。

建议保留一份只读基准配置；每个方案使用独立 JSON 和输出目录，并把数据来源、单位、插补、校准区间和参数依据写入项目记录。
