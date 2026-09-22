"""R2 城市梯度稳健性扫描（任务 C）。

原实验只有 D-HA / D-CO 两个参数域端点，无法回答审稿人会问的：
"你的结论是不是端点伪影？换个城市结构还成立吗？"

本脚本把两端点之间的连续谱离散成 5 x 5 = 25 个合成城市：
  theta_climate : 气候驱动（温度/降雨/入流序列）从 D-HA 插值到 D-CO
  theta_infra   : 城市设施参数（取水/给水/再生水能力、电网间接水强度、
                  替代率目标阈值）从 D-HA 插值到 D-CO

口径提醒（重要）：中间城市是两端点的**线性混合**，是合成谱，不是任何真实
城市。它的唯一用途是检验结论沿城市参数轴的连续性与单调性——若某个结论
只在端点成立、在中间发生跳变或非单调，则该结论不可迁移。

产出:
  city_gradient_capacity_scan.csv   25 城市 x 41 容量点的全量扫描
  city_gradient_cawcc.csv           每城市的 CAWCC(年/日) 与限制约束
  city_gradient_bottleneck.csv      每城市的瓶颈迁移链（按首次失效容量排序）

用法:
    PYTHONPATH="src;scripts" python scripts/run_r2_city_gradient.py
    PYTHONPATH="src;scripts" python scripts/run_r2_city_gradient.py --grid 3 --workers 8
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_r2 import (  # noqa: E402
    CAPACITIES_SCAN,
    annual_constraints,
    daily_constraints,
    mean_utilizations,
    peak_day_factor,
)

R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"

# 两端点之间有实际差异的城市侧参数（其余字段两端完全一致）
INFRA_PARAMS: list[tuple[str, str]] = [
    ("components.RES1.abstraction_capacity_ml_day", "水源取水能力 ML/d"),
    ("components.WTW1.daily_capacity_ml", "给水处理能力 ML/d"),
    ("components.CENTRAL_REUSE.treatment_capacity_ml_day", "再生水处理能力 ML/d"),
    ("components.WWTW1.central_reuse_fraction", "再生水中心分流比例"),
    ("components.AI_DC1.offsite_electricity_water_intensity_l_kwh", "电网间接水强度 L/kWh"),
]

_WORKER_STATE: dict = {}


def _import_engine():
    candidate_src = str(Path(sys.argv[0]).resolve().parents[1] / "src")
    candidate_scripts = str(Path(sys.argv[0]).resolve().parents[1] / "scripts")
    for _p in (candidate_src, candidate_scripts):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from aiuwm.ai_capacity import find_ai_carrying_capacity
    from aiuwm.ai_metrics import summarize_ai_water_kpis
    from aiuwm.full_engine import FullAIUWMModel

    return find_ai_carrying_capacity, summarize_ai_water_kpis, FullAIUWMModel


def _get_path(root: dict, path: str):
    target = root
    for part in path.split("."):
        target = target[int(part)] if isinstance(target, list) else target[part]
    return target


def _set_path(root: dict, path: str, value) -> None:
    parts = path.split(".")
    target = root
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[parts[-1]] = value


def blend_project(ha: dict, co: dict, theta: float) -> dict:
    """设施参数线性插值。非数值字段沿用 D-HA。"""
    trial = copy.deepcopy(ha)
    for path, _label in INFRA_PARAMS:
        low = float(_get_path(ha, path))
        high = float(_get_path(co, path))
        _set_path(trial, path, low + (high - low) * theta)
    return trial


def blend_timeseries(ts_ha: pd.DataFrame, ts_co: pd.DataFrame, theta: float) -> pd.DataFrame:
    """气候驱动线性插值：只对数值列插值，date 列保留。"""
    if theta == 0.0:
        return ts_ha.copy()
    if theta == 1.0:
        return ts_co.copy()
    mixed = ts_ha.copy()
    numeric = [c for c in ts_ha.columns if pd.api.types.is_numeric_dtype(ts_ha[c])]
    mixed[numeric] = ts_ha[numeric].astype(float) * (1.0 - theta) + ts_co[numeric].astype(float) * theta
    return mixed


def blend_constraints(c_ha: dict, c_co: dict, theta: float) -> dict:
    """再生水替代率目标阈值是城市属性，随设施轴插值；其余阈值两端一致。"""
    merged = copy.deepcopy(c_ha)
    key = "reclaimed_water_substitution_ratio"
    low = float(c_ha["constraints"][key]["value"])
    high = float(c_co["constraints"][key]["value"])
    merged["constraints"][key]["value"] = low + (high - low) * theta
    return merged


def _init_worker(ha: dict, co: dict, ts_ha: pd.DataFrame, ts_co: pd.DataFrame,
                 c_ha: dict, c_co: dict) -> None:
    _WORKER_STATE.update(ha=ha, co=co, ts_ha=ts_ha, ts_co=ts_co, c_ha=c_ha, c_co=c_co)


def _run_city(job: tuple[float, float]) -> dict:
    find_ai_carrying_capacity, summarize_ai_water_kpis, FullAIUWMModel = _import_engine()
    theta_climate, theta_infra = job
    st = _WORKER_STATE
    project = blend_project(st["ha"], st["co"], theta_infra)
    timeseries = blend_timeseries(st["ts_ha"], st["ts_co"], theta_climate)
    raw_constraints = blend_constraints(st["c_ha"], st["c_co"], theta_infra)

    rows = []
    for capacity in CAPACITIES_SCAN:
        trial = copy.deepcopy(project)
        for component in trial["components"].values():
            if component.get("kind") == "data_center":
                component["installed_it_capacity_mw"] = float(capacity)
                component.pop("capacity_schedule", None)
        result = FullAIUWMModel(trial, timeseries).run()
        record = {"ai_capacity_mw": float(capacity), **summarize_ai_water_kpis(result, trial)}
        record.update(mean_utilizations(result, trial))
        record["daily_peak_factor"] = peak_day_factor(result, trial)
        rows.append(record)
    scan = pd.DataFrame(rows)

    year_kwargs = annual_constraints(raw_constraints)
    day_kwargs = daily_constraints(raw_constraints)
    annual = find_ai_carrying_capacity(scan, year_kwargs)
    daily = find_ai_carrying_capacity(scan, day_kwargs)

    return {
        "theta_climate": theta_climate,
        "theta_infra": theta_infra,
        "scan": scan,
        "annual": annual,
        "daily": daily,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=5, help="每维取点数（默认 5 -> 25 城）")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    ha = json.loads((R2 / "ha" / "project.json").read_text(encoding="utf-8"))
    co = json.loads((R2 / "co" / "project.json").read_text(encoding="utf-8"))
    ts_ha = pd.read_csv(R2 / "ha" / "timeseries.csv")
    ts_co = pd.read_csv(R2 / "co" / "timeseries.csv")
    c_ha = json.loads((R2 / "specs" / "constraints_ha.json").read_text(encoding="utf-8"))
    c_co = json.loads((R2 / "specs" / "constraints_co.json").read_text(encoding="utf-8"))

    grid = list(np.linspace(0.0, 1.0, args.grid))
    jobs = [(float(tc), float(ti)) for tc in grid for ti in grid]
    print(f"城市梯度 {len(grid)}x{len(grid)} = {len(jobs)} 个合成城市, "
          f"每城 {len(CAPACITIES_SCAN)} 个容量点, workers={args.workers}", flush=True)

    started = time.time()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(ha, co, ts_ha, ts_co, c_ha, c_co),
    ) as pool:
        for index, item in enumerate(pool.map(_run_city, jobs), start=1):
            results.append(item)
            print(f"  [{index:2d}/{len(jobs)}] theta_c={item['theta_climate']:.2f} "
                  f"theta_i={item['theta_infra']:.2f}  "
                  f"年={item['annual']['maximum_safe_ai_capacity_mw']:.0f} "
                  f"日={item['daily']['maximum_safe_ai_capacity_mw']:.0f}  "
                  f"{time.time() - started:6.1f}s", flush=True)

    # ---- 1) 全量容量扫描 ----------------------------------------------------
    scan_frames = []
    for item in results:
        frame = item["scan"].copy()
        frame.insert(0, "theta_infra", item["theta_infra"])
        frame.insert(0, "theta_climate", item["theta_climate"])
        scan_frames.append(frame)
    full_scan = pd.concat(scan_frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    full_scan.to_csv(OUT / "city_gradient_capacity_scan.csv", index=False, encoding="utf-8-sig")

    # ---- 2) 每城 CAWCC 汇总 -------------------------------------------------
    summary = pd.DataFrame([{
        "theta_climate": r["theta_climate"],
        "theta_infra": r["theta_infra"],
        "CAWCC_年_MW": r["annual"]["maximum_safe_ai_capacity_mw"],
        "年尺度限制约束": r["annual"]["limiting_constraint"],
        "CAWCC_日_MW": r["daily"]["maximum_safe_ai_capacity_mw"],
        "日尺度限制约束": r["daily"]["limiting_constraint"],
    } for r in results])
    summary.to_csv(OUT / "city_gradient_cawcc.csv", index=False, encoding="utf-8-sig")

    # ---- 3) 瓶颈迁移链 ------------------------------------------------------
    bottleneck_rows = []
    for item in results:
        for scale, block in (("年", item["annual"]), ("日", item["daily"])):
            boundaries = block.get("constraint_boundaries") or block.get("boundaries") or {}
            if isinstance(boundaries, dict):
                ranked = sorted(
                    ((k, v.get("first_failed_capacity_mw") if isinstance(v, dict) else v)
                     for k, v in boundaries.items()),
                    key=lambda kv: (kv[1] is None, kv[1]),
                )
                for rank, (name, first_fail) in enumerate(ranked, start=1):
                    bottleneck_rows.append({
                        "theta_climate": item["theta_climate"],
                        "theta_infra": item["theta_infra"],
                        "尺度": scale,
                        "瓶颈位次": rank,
                        "约束": name,
                        "首次失效容量_MW": first_fail,
                    })
    if bottleneck_rows:
        pd.DataFrame(bottleneck_rows).to_csv(
            OUT / "city_gradient_bottleneck.csv", index=False, encoding="utf-8-sig"
        )

    elapsed = time.time() - started
    print(f"\n完成 {len(results)} 城 / {len(full_scan):,} 行扫描 / {elapsed:.0f}s")
    print(summary.pivot(index="theta_climate", columns="theta_infra",
                        values="CAWCC_日_MW").to_string(float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main()
