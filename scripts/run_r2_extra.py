"""R2 扩展块: 城市侧参数敏感性 + 能源富集补充情景.

新增动机（nature-skills 方案审查后补的两个方法学缺口）:

1. 原敏感性分析只覆盖 AI 侧 6 个参数，而头号结论"承载容量上限由水源取水与
   给水处理决定"恰恰建立在城市侧【设定】参数之上。城市侧参数从未做过不确定
   性检验，结论对设定值的稳健性未知。此处补 Morris + Sobol。
   观测指标选 `max_source_abstraction_ratio` 与 `max_wtw_utilization`，因为
   它们正是年尺度与日尺度 CAWCC 的限制约束——直接决定承载容量。

2. 能源富集补充情景（方案 4.3）定义了却从未执行（13.1 命令序列与 13.2 产物
   表均无它），属悬空目标。此处补跑并与常规情景对比。

用法:
    python scripts/run_r2_extra.py                       # 全部
    python scripts/run_r2_extra.py --blocks city         # 仅城市侧敏感性
    python scripts/run_r2_extra.py --blocks energy       # 仅能源富集
    python scripts/run_r2_extra.py --sobol-samples 128   # 提高 Sobol 样本量
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiuwm.ai_capacity import find_ai_carrying_capacity
from aiuwm.ai_metrics import summarize_ai_water_kpis
from aiuwm.full_engine import FullAIUWMModel
from aiuwm.sensitivity import morris_sensitivity, sobol_sensitivity

# 复用 run_r2.py 的口径定义，保证与本轮主实验完全一致
from run_r2 import (
    CAPACITIES_SCAN,
    annual_constraints,
    daily_constraints,
    scan_with_mean,
)

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"

# 城市侧参数: 路径 -> (相对基线的下界系数, 上界系数, 中文说明)
# 选择标准: 这些参数决定城市侧约束的松紧，也就是 CAWCC 的限制约束取值。
CITY_PARAM_SPEC: dict[str, tuple[float, float, str]] = {
    "components.RES1.abstraction_capacity_ml_day": (0.75, 1.25, "水源取水能力"),
    "components.WTW1.daily_capacity_ml": (0.75, 1.25, "给水处理能力"),
    "components.WWTW1.daily_capacity_ml": (0.75, 1.25, "污水处理能力"),
    "components.CENTRAL_REUSE.treatment_capacity_ml_day": (0.75, 1.25, "再生水处理能力"),
    "components.DM1.leakage_fraction": (0.50, 1.50, "配水漏损率"),
    "local_areas.LA1.demand_profiles.0.base_value": (0.75, 1.25, "居民用水定额"),
    "components.RES1.capacity_ml": (0.75, 1.25, "水库库容"),
}

# 观测指标: 年尺度与日尺度 CAWCC 的限制约束
CITY_METRICS = {
    "max_source_abstraction_ratio": "年尺度头号约束（水源取水占用率）",
    "max_wtw_utilization": "日尺度头号约束（给水处理峰值利用率）",
}

SEED = 20260918


def _get_path(root: dict, path: str):
    target = root
    for part in path.split("."):
        target = target[int(part)] if isinstance(target, list) else target[part]
    return target


def _set_path(root: dict, path: str, value) -> None:
    target = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[parts[-1]] = value


def make_evaluator(project: dict, timeseries: pd.DataFrame, metric: str):
    def evaluator(parameters: dict) -> float:
        trial = copy.deepcopy(project)
        for path, value in parameters.items():
            _set_path(trial, str(path), float(value))
        result = FullAIUWMModel(trial, timeseries).run()
        return float(summarize_ai_water_kpis(result, trial)[metric])

    return evaluator


def city_bounds(project: dict) -> dict[str, tuple[float, float]]:
    """按各域自身基线生成扰动区间（不跨域共用一组固定数值）。"""
    bounds: dict[str, tuple[float, float]] = {}
    for path, (lo_f, hi_f, _label) in CITY_PARAM_SPEC.items():
        base = float(_get_path(project, path))
        bounds[path] = (round(base * lo_f, 6), round(base * hi_f, 6))
    return bounds


def run_city_sensitivity(domain: str, trajectories: int, sobol_samples: int) -> None:
    folder = OUT / domain
    folder.mkdir(parents=True, exist_ok=True)
    project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(R2 / domain / "timeseries.csv")

    trial = copy.deepcopy(project)
    trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = 1000.0
    bounds = city_bounds(trial)

    print(f"[{domain}] 城市侧参数基线:", flush=True)
    for path, (lo, hi) in bounds.items():
        print(f"    {CITY_PARAM_SPEC[path][2]:12s} {path:52s} {lo:>10.3f} ~ {hi:>10.3f}", flush=True)

    morris_rows = []
    for metric, desc in CITY_METRICS.items():
        print(f"[{domain}] morris: {metric} ({desc}) ...", flush=True)
        res = morris_sensitivity(
            make_evaluator(trial, timeseries, metric), bounds,
            trajectories=trajectories, seed=SEED,
        )
        res = res.copy()
        res.insert(0, "metric", metric)
        res["metric_desc"] = desc
        morris_rows.append(res)
    morris = pd.concat(morris_rows, ignore_index=True)
    morris.to_csv(folder / "city_morris.csv", index=False)

    # Sobol 只跑年尺度头号约束（最贵，且它就是年尺度 CAWCC 的限制约束）
    metric = "max_source_abstraction_ratio"
    print(f"[{domain}] sobol: {metric} (N={sobol_samples}) ...", flush=True)
    sobol = sobol_sensitivity(
        make_evaluator(trial, timeseries, metric), bounds,
        samples=sobol_samples, seed=SEED,
    )
    sobol = sobol.copy()
    sobol.insert(0, "metric", metric)
    sobol.to_csv(folder / "city_sobol.csv", index=False)

    print(f"\n=== [{domain}] 城市侧 Morris (mu* 排序) ===")
    for metric in CITY_METRICS:
        sub = morris[morris["metric"] == metric].sort_values("mu_star", ascending=False)
        print(f"-- {metric} --")
        print(sub.to_string(index=False))
    print(f"=== [{domain}] 城市侧 Sobol (ST 排序) ===")
    print(sobol.sort_values("ST", ascending=False).to_string(index=False))


def run_energy_rich(domains: tuple[str, ...] = ("ha", "co")) -> None:
    """能源富集补充情景，与常规情景对比 CAWCC 与瓶颈。

    方案 v3 只把它定义在 D-HA 上叠加，但 D-HA 的瓶颈在水侧（9.5 节迁移链里
    电力排第 3 位、1450 MW 才失效），在那里放宽电力约束本就不会有反应。
    真正电力先到顶的是 **D-CO**（迁移链第 1 位即接网容量 1500 MW），
    因此 v4 把该对照扩展到两端点，否则这个情景检验不到它想检验的东西。
    """
    for domain in domains:
        _run_energy_rich_one(domain)


def _run_energy_rich_one(domain: str) -> None:
    folder = OUT / domain
    folder.mkdir(parents=True, exist_ok=True)
    timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
    constraints = json.loads((R2 / "specs" / f"constraints_{domain}.json").read_text(encoding="utf-8"))

    base = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
    rich: dict
    if (R2 / domain / "energy_rich.json").exists():
        rich = json.loads((R2 / domain / "energy_rich.json").read_text(encoding="utf-8"))
    else:
        # D-CO 未预生成，按同一 overlay 语义构造
        rich = copy.deepcopy(base)
        dc = rich["components"]["AI_DC1"]
        dc["grid_connection_capacity_mw"] = 2600.0
        dc["offsite_electricity_water_intensity_l_kwh"] = 0.08

    variants = {"常规": base, "能源富集": rich}

    rows = []
    for label, project in variants.items():
        scan = scan_with_mean(project, timeseries, CAPACITIES_SCAN)
        positive = scan[scan["ai_capacity_mw"] >= 50].reset_index(drop=True)

        daily = find_ai_carrying_capacity(positive, {"constraints": daily_constraints(constraints)})
        annual = find_ai_carrying_capacity(positive, {"constraints": annual_constraints(constraints)})

        rows.append({
            "情景": label,
            "接网容量_MW": project["components"]["AI_DC1"]["grid_connection_capacity_mw"],
            "电网间接水强度_L_per_kWh": project["components"]["AI_DC1"].get(
                "offsite_electricity_water_intensity_l_kwh"
            ),
            "CAWCC_年_MW": annual.get("maximum_safe_ai_capacity_mw"),
            "年尺度限制约束": annual.get("limiting_constraint"),
            "CAWCC_日_MW": daily.get("maximum_safe_ai_capacity_mw"),
            "日尺度限制约束": daily.get("limiting_constraint"),
        })
        print(f"[{label}] 年 {annual.get('maximum_safe_ai_capacity_mw')} MW "
              f"/ {annual.get('limiting_constraint')}; "
              f"日 {daily.get('maximum_safe_ai_capacity_mw')} MW "
              f"/ {daily.get('limiting_constraint')}", flush=True)

    frame = pd.DataFrame(rows)
    frame.insert(0, "域", domain.upper())
    frame.to_csv(folder / "energy_rich_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 能源富集补充情景（D-{domain.upper()}）===")
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--blocks", default="city,energy")
    parser.add_argument("--trajectories", type=int, default=24)
    parser.add_argument("--sobol-samples", type=int, default=64)
    args = parser.parse_args()

    blocks = {b.strip() for b in args.blocks.split(",")}

    if "city" in blocks:
        for domain in args.domains.split(","):
            run_city_sensitivity(domain.strip(), args.trajectories, args.sobol_samples)

    if "energy" in blocks:
        run_energy_rich()

    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
