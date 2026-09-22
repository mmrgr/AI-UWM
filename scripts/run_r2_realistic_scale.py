"""R2 现实尺度诊断：约束在什么**现实**配置下才真正触发？

**动机（用户质疑"感觉还是不太对劲"→ 审计证实）**
`audit_experiment_design.py` 发现：在现实的 50–300 MW 园区区间内，
两域所有承载约束的最大取值都 < 0.95（HA 0.83 / CO 0.78），
即**现实园区尺度下承载约束根本不触发**。而全文所有主结论都建立在
1150 / 1450 MW 这个不现实尺度上。文档只写了"要折算到 50–300 MW"，
但**折算后的结论从未被真正计算过**。

本脚本直接补这一块：把 AI 装机拉到现实区间，同时沿**城市规模轴**扫描，
找出"多大的城市 + 多大的园区"才会真正触及水系统承载边界。

**设计**
- 城市规模：20 / 35 / 50 / 75 / 100 / 150 / 200 万人（设施能力与需求按人口线性缩放）
- AI 装机：0 … 600 MW，步长 25 MW（覆盖并超出真实园区区间）
- 观测：每条约束在**全期**的最大取值，以及首次达到阈值 0.95（或违反 ≥ 阈值）的容量
- 输出：二维表 + 「临界装机」曲线，直接给出结论的现实适用区间

**缩放规则（重要，须写进论文）**
`local_areas.LA1.base_population` 与全部容量型字段（`*_ml`、`*_ml_day`、
`daily_capacity_ml`、`treatment_capacity_ml_day`、含水厂/污水厂/再生水/管网/库容）
按 `pop / 1e6` 等比缩放；`demand_profiles` 中 `l_capita_day` 类按人口、其余同样等比缩放。
这样做的含义是"把同一座城市整体缩小/放大"，而不是"往同一座城市里塞不同的 AI"——
后者才是 0–2000 MW 压力测试网格在做的事。

用法:
    PYTHONPATH="src;scripts" python scripts/run_r2_realistic_scale.py
    PYTHONPATH="src;scripts" python scripts/run_r2_realistic_scale.py --populations 0.5,1.0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aiuwm.ai_metrics import summarize_ai_water_kpis  # noqa: E402
from aiuwm.full_engine import FullAIUWMModel  # noqa: E402
from run_r2 import (  # noqa: E402
    R2,
    daily_constraints,
    mean_utilizations,
)

OUT = ROOT / "validation_artifacts" / "r2"

DOMAIN_LABEL = {"ha": "D-HA 缺水—高冷负荷端", "co": "D-CO 气候凉爽端"}

# 会被人口规模缩放的能力型字段（单位含 ML 的容量/速率，以及日能力）
SCALABLE_KEYS = (
    "capacity_ml", "initial_ml", "inflow_ml", "daily_capacity_ml",
    "abstraction_capacity_ml_day", "treatment_capacity_ml_day",
    "installed_it_capacity_mw",  # 不缩放，单独处理
)


def _scale_project(project: dict, factor: float) -> dict:
    """把整座城市按人口因子等比缩放（AI 装机除外，它由调用方设定）。"""
    cfg = copy.deepcopy(project)

    # 人口与需求
    for area in cfg["local_areas"].values():
        area["base_population"] = float(area["base_population"]) * factor
        for prof in area.get("demand_profiles", []):
            if prof.get("unit") == "l_capita_day":
                continue  # 人均定额不随人口变
            prof["base_value"] = float(prof["base_value"]) * factor

    # 组件能力（跳过 AI 数据中心，它的装机由扫描给定）
    for comp in cfg["components"].values():
        if comp.get("kind") == "data_center":
            continue
        for key in SCALABLE_KEYS:
            if key in comp and isinstance(comp[key], (int, float)):
                comp[key] = float(comp[key]) * factor
    return cfg


def _constraint_max(cfg: dict, timeseries: pd.DataFrame, constraints: dict) -> dict:
    result = FullAIUWMModel(copy.deepcopy(cfg), timeseries).run()
    kpi = summarize_ai_water_kpis(result, cfg)
    util = mean_utilizations(result, cfg)
    record = {**kpi, **util}
    day = daily_constraints(constraints)
    return {k: record.get(k) for k in day if k in record}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--populations", default="0.2,0.35,0.5,0.75,1.0,1.5,2.0",
                        help="人口（百万人），逗号分隔")
    parser.add_argument("--capacities", default="0,25,50,...,600")
    parser.add_argument("--domains", default="ha,co")
    args = parser.parse_args()

    populations = [float(x) for x in args.populations.split(",") if x.strip()]
    capacities = list(np.arange(0, 601, 25, dtype=float))
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    rows: list[dict] = []
    for domain in domains:
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
        constraints = json.loads(
            (R2 / "specs" / f"constraints_{domain}.json").read_text(encoding="utf-8")
        )
        thresholds = {k: v.get("value") for k, v in constraints["constraints"].items()}
        thresholds["system_reliability_fraction"] = thresholds.get("system_reliability_fraction", 1.0)
        print(f"=== {domain.upper()} 阈值: {json.dumps(thresholds, ensure_ascii=False)}", flush=True)

        for pop in populations:
            factor = pop  # 以 100 万为基准
            base = _scale_project(project, factor)
            for cap in capacities:
                cfg = copy.deepcopy(base)
                for comp in cfg["components"].values():
                    if comp.get("kind") == "data_center":
                        comp["installed_it_capacity_mw"] = float(cap)
                        comp.pop("capacity_schedule", None)
                metrics = _constraint_max(cfg, timeseries, constraints)
                rows.append({
                    "域": domain,
                    "域标签": DOMAIN_LABEL[domain],
                    "人口_万人": pop * 100.0,
                    "AI装机_MW": float(cap),
                    **{f"{k}": v for k, v in metrics.items()},
                })
            print(f"  [{domain}] 人口 {pop * 100:.0f} 万 完成（{len(capacities)} 个容量点）", flush=True)

    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "realistic_scale_scan.csv", index=False, encoding="utf-8-sig")

    # ---- 逐域逐人口求「首次触及阈值」的装机 ---------------------------------
    # 规划裕度型（<0.95）：首次 ≥0.95；比例型（≥阈值）：首次 <阈值
    margin_keys = [c for c in frame.columns if c.startswith("max_") or c == "peak_capacity_ratio"]
    margin_keys = [c for c in margin_keys if c not in
                   ("max_source_abstraction_ratio", "max_daily_average_grid_connection_ratio",
                    "max_local_connection_ratio", "peak_capacity_ratio")]
    upper_keys = [c for c in frame.columns
                  if c in ("max_source_abstraction_ratio", "max_daily_average_grid_connection_ratio",
                           "max_local_connection_ratio", "peak_capacity_ratio")]
    lower_keys = [c for c in frame.columns
                  if c in ("reclaimed_water_substitution_ratio", "system_reliability_fraction")]

    first_rows = []
    for (domain, pop), group in frame.groupby(["域", "人口_万人"]):
        group = group.sort_values("AI装机_MW")
        record = {"域": domain, "人口_万人": pop, "人口_万人_显示": f"{pop:.0f}"}
        for key in margin_keys + upper_keys:
            hits = group[group[key] >= 0.95]
            record[f"首个{key}_≥0.95_MW"] = (
                float(hits.iloc[0]["AI装机_MW"]) if not hits.empty else None
            )
        for key in lower_keys:
            thr = thresholds.get(key)
            if thr is None:
                continue
            hits = group[group[key] < float(thr)]
            record[f"首个{key}_<{thr}_MW"] = (
                float(hits.iloc[0]["AI装机_MW"]) if not hits.empty else None
            )
        first_rows.append(record)

    first = pd.DataFrame(first_rows)
    first.to_csv(OUT / "realistic_scale_first_binding.csv", index=False, encoding="utf-8-sig")

    print("\n=== 现实尺度下各约束的最大取值（跨全部容量点与人口）===")
    show = [c for c in frame.columns if c.startswith("max_") or c == "peak_capacity_ratio"
            or c in ("reclaimed_water_substitution_ratio", "system_reliability_fraction")]
    print(frame.groupby(["域", "人口_万人"])[show].max().round(3).to_string())

    print("\n=== 首个触及阈值 0.95 的 AI 装机（None = 600 MW 内都不触发）===")
    cols = ["域", "人口_万人_显示"] + [c for c in first.columns if c.startswith("首个")]
    print(first[cols].to_string(index=False))
    print(f"\n-> {OUT / 'realistic_scale_scan.csv'}")
    print(f"-> {OUT / 'realistic_scale_first_binding.csv'}")


if __name__ == "__main__":
    main()
