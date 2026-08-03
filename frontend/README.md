# WaterMet² Studio 中文使用教程

这份说明面向“手里只有业务数据、没有编程经验”的用户。照着第 1～8 步操作，即可完成数据准备、模型配置、检查、运行和结果导出。

WaterMet² 是日尺度城市水系统战略模型，不是管网压力模型。它回答的是：

- 城市每天需要多少水，能够供应多少，缺水多少；
- 水源、给水厂、管网、用户、排水、污水厂和回用系统之间有多少水流；
- 系统消耗多少电力、燃料、材料和药剂；
- 产生多少温室气体、酸化、富营养化和成本；
- 不同建设或节水方案在长期运行中哪个更好。

## 0. 开始前必须知道

只有一张 CSV 还不足以建立可信模型。至少还需要知道系统的基本结构和参数，例如：

- 有几个水源、给水厂、水池、配水系统、排水系统和污水厂；
- 各设施容量、初始储量、漏损率、处理损失率；
- 服务人口、区域面积和人均用水量；
- CSV 每一列的含义和单位。

如果暂时没有全部参数，可以先复制完整示例，用已有数据替换气象、水源入流和人口；未知参数保留示例值进行试跑，但正式研究前必须用实测资料或文献值替换，并记录来源。

## 1. 准备软件

需要安装：

- Python 3.10 或更高版本；
- Node.js 20 或更高版本。

在项目目录 `D:\研二上\我的论文\watermet2` 打开 PowerShell：

```powershell
python -m pip install -e ".[studio,test]"
Set-Location frontend
npm install
```

## 2. 启动系统

打开两个 PowerShell 窗口。

窗口一，在项目根目录启动 WaterMet² 模型 API：

```powershell
watermet2-api
```

如果提示找不到 `watermet2-api`，使用：

```powershell
python -m uvicorn watermet2_repro.api:app --host 127.0.0.1 --port 8000
```

窗口二启动网页：

```powershell
Set-Location "D:\研二上\我的论文\watermet2\frontend"
npm run dev
```

打开浏览器访问：

<http://127.0.0.1:5173>

页面会自动载入一个 731 天完整示例。先不要修改示例，点击右上角“数据检查”和“运行仿真”，确认电脑环境能够正常得到结果。

## 3. 整理自己的文件

建议新建一个独立目录：

```text
my_watermet_case/
├─ project.json
├─ timeseries.csv
├─ 数据来源说明.txt
└─ 输出结果/
```

从以下位置复制项目模板：

```text
examples/demo_full/project.json
```

CSV 第一行是列名，每一行代表一天。最简单的形式如下：

```csv
date,rainfall_mm,temperature_c,source_inflow_ml,population
2020-01-01,3.2,5.6,210.4,500000
2020-01-02,0.0,4.9,198.1,500010
2020-01-03,8.1,4.2,225.7,500020
```

必须满足：

1. `date` 格式为 `YYYY-MM-DD`；
2. 模拟期每天恰好一行，不能缺日、不能重复；
3. 数值列不要包含单位文字、破折号或空字符串；
4. CSV 列名必须和 `project.json` 引用的名称一致；
5. 文件保存为 UTF-8 CSV。

常用单位：

| 数据 | 单位 |
|---|---|
| 水源入流、需水量、处理能力 | ML/day |
| 水库、水池、调蓄设施储量 | ML |
| 降雨、蒸发 | mm/day |
| 温度 | °C |
| 人口 | 人 |
| 污染物负荷 | kg/day |
| 电力 | kWh |
| 成本 | EUR |

注意：`1 ML = 1000 m³`。把 m³ 当成 ML 会导致结果放大 1000 倍。

## 4. 配置项目 JSON

用记事本、VS Code 或其他文本编辑器打开 `project.json`。

### 4.1 设置模拟日期

```json
"simulation": {
  "start": "2020-01-01",
  "end": "2024-12-31"
}
```

日期范围必须被 CSV 完整覆盖。

### 4.2 让水源读取 CSV

```json
"WR1": {
  "kind": "water_resource",
  "capacity_ml": 12000,
  "initial_ml": 6000,
  "inflow_column": "source_inflow_ml",
  "abstraction_capacity_ml_day": 300
}
```

`inflow_column` 必须等于 CSV 中的列名。如果有两个水源，可分别使用 `wr1_inflow_ml`、`wr2_inflow_ml`。

### 4.3 配置城市区域

```json
"LA1": {
  "base_population": 500000,
  "area_ha": 8000,
  "weather_columns": {
    "precipitation": "rainfall_mm",
    "temperature": "temperature_c"
  },
  "demand_profiles": [
    {
      "name": "domestic",
      "base_value": 150,
      "unit": "l_capita_day",
      "return_fraction": 0.9
    }
  ]
}
```

如果 CSV 每天有人口数据，可使用：

```json
"population_column": "population"
```

### 4.4 检查供水路径

一条典型路径为：

```text
水源 → 原水输水 → 给水厂 → 清水干管 → 服务水池 → 配水系统 → 城市区域
```

如果同一区域由两条路径供水，`allocation` 必须合计为 1，例如 0.9 + 0.1。

复杂的污染物、资产、材料、药剂、成本、回用和干预参数，可从 `examples/demo_full/project.json` 中保留并逐项修改。根目录 [README.md](../README.md) 有全部字段说明。

## 5. 在网页中输入项目和时间序列

### 5.1 打开项目

1. 点击顶部“打开”；
2. 选择自己的 `project.json`；
3. 页面切换到“系统模型”；
4. 确认画布上的节点和连线与真实系统一致。

连线箭头表示水流方向。运行前，线上显示“配比”；运行后，线上显示整个模拟期的平均流量，单位为 `ML/d`。

### 5.2 输入 CSV

1. 点击左侧“数据中心”；
2. 点击“输入时间序列 CSV”；
3. 选择自己的 `timeseries.csv`；
4. 查看记录数、字段数和缺失值；
5. 在“列映射”中确认常用字段；
6. 自定义水源列应选择“保留原列名”；
7. 点击“应用映射”；
8. 用右侧曲线确认数据量级和变化趋势合理。

“导出当前 CSV”可以把网页中当前使用的数据重新下载，用于确认映射后的列名。

## 6. 运行检查

点击右上角“数据检查”。系统会依次检查：

- 项目是否存在水源和配水终点；
- 是否存在孤立节点；
- 设施容量、区域人口和面积是否缺失；
- 供水或污水分配比例是否正确；
- CSV 是否为空；
- JSON 引用的 CSV 列是否存在；
- 日期格式是否正确；
- 是否有重复日期和缺失日期；
- 必需列是否含空值或非数值；
- Python 引擎能否通过最终项目校验。

底部面板显示错误、警告和运行日志。错误必须修复后才能运行。警告应核实，但不一定阻止运行。

常见错误：

| 提示 | 处理方法 |
|---|---|
| 缺少时间序列列 | 修改 CSV 列名，或修改 JSON 中的 `inflow_column`/`weather_columns` |
| 时间序列缺日 | 补齐从开始日至结束日的每日记录 |
| 日期重复 | 删除或合并同一天的重复行 |
| 分配比例不为 1 | 修改连线分配比例，或点击“一键归一化” |
| 孤立节点 | 在画布中补上正确连线，或删除不参与方案的组件 |
| 容量参数缺失 | 选中节点，在右侧属性面板填写容量 |

## 7. 运行模拟

1. 点击左侧“仿真运行”；
2. 再次确认开始日期、结束日期、数据行数和组件数；
3. 点击“后端校验”；
4. 校验通过后点击“开始仿真”，也可直接点击右上角“运行仿真”；
5. 等待系统自动进入“结果分析”。

运行完成后返回“系统模型”，每条线上会显示模拟期平均流量，例如：

```text
194.10 ML/d
```

这是模拟流量，不是路径分配比例。

## 8. 查看和导出结果

结果页顶部显示：

- 供水可靠率；
- 总缺水量；
- 总能耗；
- 净温室气体；
- 现值总成本；
- 23 类风险。

下方包括水代谢桑基图、风险排序、供需和储量时间序列，以及完整结果表。

“完整结果表”可以选择并导出：

| 结果表 | 含义 |
|---|---|
| `system_daily` | 全系统每日供水、损失、储量、能耗、环境影响和成本 |
| `subcatchment_daily` | Subcatchment 每日汇总 |
| `area_daily` | 城市区域需求、供水、回用、污水和径流 |
| `indoor_daily` | 室内/房屋类型的需求和供水 |
| `component_daily` | 每个设施的流入、流出、储量、能耗、药剂和成本 |
| `pollutant_daily` | 每个组件、出流类型和污染物的质量通量 |
| `recovery_daily` | 沼气、电、热、肥料等资源回收 |
| `material_events` | 管道建设、修复和材料事件 |
| `asset_daily` | 资产失效、停运、维护和维修成本 |
| `flood_daily` | 溢流、淹没面积、流速和危险面积 |
| `risk_daily` | 23 类风险的每日指标 |
| `risk_summary` | 每类风险的概率、严重度和累计分数 |

导出方法：

- “导出当前表 CSV”：下载当前选择的完整日尺度表；
- “导出全部结果 ZIP”：重新执行一次校验和模拟，下载全部日/月/年 CSV、`summary.json` 和运行时 `project.json`。

ZIP 是论文分析和归档推荐使用的完整结果。

## 9. 比较不同方案

1. 先运行基准方案；
2. 进入“情景对比”，输入“基准方案”，保存当前结果；
3. 修改漏损率、容量、回用比例或干预日期；
4. 重新运行；
5. 保存为“节水方案”“新增水源方案”等；
6. 比较可靠率、缺水、GHG 和成本。

校准、Monte Carlo、不确定性、Compromise Programming 和 Pareto 优化属于批量长任务，入口和命令在“高级分析”页面及根目录 README 中。这些计算仍调用同一个模型引擎。

## 10. 如何判断结果是否可信

至少完成以下检查：

1. 供水量级是否接近历史统计；
2. `water_demand_ml = delivered_total_ml + unmet_demand_ml`；
3. 水源、给水厂、污水厂的日流量是否超过配置能力；
4. 校准和验证使用不同年份；
5. 记录每个参数的来源和单位；
6. 对漏损率、人口增长、能耗和排放因子做不确定性分析；
7. 不把本模型结果解释为管网节点压力或二维积水深度。

## 11. 开发者验证命令

```powershell
Set-Location frontend
npm run lint
npm test -- --run
npm run build
Set-Location ..
python -m pytest -q
python scripts\run_full_validation.py
```

如果页面提示 API 未连接，先访问 <http://127.0.0.1:8000/api/health>。正常应返回：

```json
{"status":"ok","engine":"watermet2-reproduction"}
```
