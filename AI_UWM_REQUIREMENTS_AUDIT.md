# AI-UWM 与开题报告功能核查（当前版本）

核查对象：`C:\Users\mmrgr\Desktop\开题\开题报告3.2.docx` 与当前工作树。

## 已实现的研究工作流

| 开题要求 | 当前实现 |
|---|---|
| AI—城市供排水耦合 | `data_center.py` + `full_engine.py`：IT 容量、负荷、PUE、热负荷、湿/干/混合冷却、蒸发、漂水、排污、补水、回流、污水和回用。 |
| 守恒闭合 | `data_center_daily.water_balance_residual_ml`；示例最大绝对残差 `8.88e-16 ML`。 |
| 年/日尺度 | daily/weekly/monthly/annual 输出和聚合测试。 |
| 容量与五维情景 | `ai_scenarios.py`：容量、冷却、水源、水文气候、基础设施。 |
| CAWCC 扫描 | `scan_ai_capacity`、`find_ai_carrying_capacity`、非单调可行区间和逐约束边界。 |
| 瓶颈与干预 | `evaluate_ai_interventions`、`summarize_constraint_boundaries`、`ai-bottlenecks`。 |
| S0–S3/G0–G3 | `research.py`：能力余量和气候压力倍率、矩阵运行、CSV 输出。 |
| AI/普通工业对照 | `industrial_control_project`、`compare_ai_industrial`、`ai-industrial`。 |
| 小时压力边界 | `hourly_stress_scan`：训练/推理 profile、峰值因子、相位参数、小时峰值容量比和无量纲裕度。 |
| 不确定性与稳健性 | Morris、Sobol、Monte Carlo、`robustness_matrix`、超限概率。 |
| 参数证据链 | `parameter_provenance`：路径、单位、来源、年份、证据类型、不确定性。 |
| 能—碳后评估 | 系统能耗、碳排放、间接电力水足迹；不进入水承载失效判据。 |
| API/CLI/前端客户端 | 新增研究 API、CLI 命令和 `frontend/src/services/api.ts` 类型化调用。 |

## 本轮新增接口

```text
POST /api/ai/hourly-boundary
POST /api/ai/states
POST /api/ai/industrial-control
POST /api/ai/robustness
POST /api/ai/provenance
```

```text
watermet2 ai-states
watermet2 ai-industrial
watermet2 ai-hourly-boundary
watermet2 ai-robustness
watermet2 ai-provenance
```

## 当前仍需目标城市数据完成的部分

1. 用目标城市逐小时气象、AI 负荷、城市需水和设施能力替换典型 profile/倍率。
2. 将 WTW、WWTW、再生水源、深度处理、再生水输配、局部接入和电网容量标定为现场值。
3. 使用实测 AI 负荷、WUE、PUE、补水和回流数据完成校准—验证拆分。
4. 根据工程设计步长完成干预边际曲线和步长收敛分析。
5. 前端目前提供研究 API 客户端，专用瓶颈图、边界图和证据表页面仍可继续补充。

## 验证结果

```text
python -m pytest -q
58 passed in 5.47s
Exit status: 0

npm test -- --run       # frontend
6 passed
Exit status: 0

npm run lint            # frontend
通过
Exit status: 0

npm run build           # frontend
✓ 1034 modules transformed
✓ built in 4.04s
Exit status: 0
```

CLI 已实际运行 `ai-states`、`ai-industrial`、`ai-provenance`，均生成结果文件；API 端点由 Python 测试覆盖。
