# AI算力—城市水承载力扩展验证报告

验证日期：2026-08-20。验证对象：`examples/ai_data_center/project.json`。

## 精确验证命令与结果

```powershell
python -m pytest -q
# 49 passed in 4.90s

python scripts\run_full_validation.py
# 29/29 全部通过

python scripts\run_ai_validation.py
# all_checks_passed: true
# maximum_safe_ai_capacity_mw: 1400.0
# first_failed_capacity_mw: 1500.0
# limiting_constraint: domestic_unmet_ml

Set-Location frontend
npm test -- --run
# 3 test files, 6 tests passed
npm run build
# TypeScript及Vite生产构建成功
```

## A–O验收

| 条目 | 结果 | 验证依据 |
|---|---|---|
| A 原功能不回退 | 通过 | Python 49项测试；旧全流程29/29 |
| B AI质量守恒 | 通过 | 最大日残差 `<1e-9 ML` |
| C 0–2 GW扫描 | 通过 | `ai_capacity_scan.csv`，100 MW步长 |
| D 六类冷却 | 通过 | 单元测试及场景矩阵 |
| E potable+reclaimed | 通过 | 示例两类实际供水均大于0 |
| F 再生水容量约束 | 通过 | reuse utilization不超过1，专门集成测试 |
| G blowdown进入WWTW | 通过 | AI TDS及下游污水流量测试 |
| H 三种水账户 | 通过 | Withdrawal/Consumption/Return Flow逐日输出 |
| I 峰值分析 | 通过 | 最大日、P95、最大7日、夏季峰值 |
| J Baseline vs AI | 通过 | `ai_baseline_delta.csv` |
| K 最大安全容量 | 通过 | 示例阈值区间1400–1500 MW |
| L 概率承载边界 | 通过 | `ai_probabilistic_threshold.csv` |
| M Morris/Sobol | 通过 | `ai_morris.csv`、`ai_sobol.csv` |
| N Studio | 通过 | AI节点、属性、AI Water图表；6项测试与构建通过 |
| O 文档与示例 | 通过 | `AI_DATA_CENTER_RESEARCH_CN.md`及完整示例 |

## 示例结果说明

当前十日高温/枯水示例的确定性阈值为 **1400 MW安全、1500 MW首次失败**，限制项是居民缺水。该数字仅用于证明阈值算法和完整研究链可重复运行，不代表任何真实城市的许可容量；研究应用必须替换本地设施、气象、入流、水质、电网和运行参数。

机器可读判据见 `ai_validation_report.json`。其余CSV提供Baseline差分、容量曲线、阈值分类、设施日/月/年最大利用率、Pareto策略及敏感性结果。
