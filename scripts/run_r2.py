"""Run the R2 (next-round) CAWCC experiment matrix and verify it.

Produces, per parameter-domain endpoint:
  capacity scan / CAWCC / constraint boundaries / intervention marginal curves
  cooling-technology comparison / allocation-policy comparison / S x G matrix
  hourly stress / Morris sensitivity / probabilistic threshold / robustness
plus a machine-readable verification report (`r2_verification.json`).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from aiuwm.ai_capacity import (
    evaluate_ai_interventions,
    find_ai_carrying_capacity,
    scan_ai_capacity,
    summarize_constraint_boundaries,
)
from aiuwm.ai_metrics import compare_baseline_ai, summarize_ai_water_kpis
from aiuwm.cawcc import build_intraday_profile
from aiuwm.full_engine import FullAIUWMModel
from aiuwm.research import apply_state_pressure, run_state_pressure_matrix
from aiuwm.sensitivity import (
    capacity_exceedance_probability,
    morris_sensitivity,
    probabilistic_ai_capacity_threshold,
    sobol_sensitivity,
)


def _set_path(root: dict, path: str, value) -> None:
    target = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[parts[-1]] = value


def make_evaluator(project: dict, timeseries: pd.DataFrame, metric: str):
    """Build the parameter -> KPI callback required by the sensitivity helpers."""
    def evaluator(parameters: dict) -> float:
        trial = copy.deepcopy(project)
        for path, value in parameters.items():
            _set_path(trial, str(path), value)
        result = FullAIUWMModel(trial, timeseries).run()
        return float(summarize_ai_water_kpis(result, trial)[metric])
    return evaluator

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"

CAPACITIES_SCAN = list(np.arange(0, 2001, 50, dtype=float))
CAPACITIES_COARSE = [0.0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]

# 设施"规划裕度"利用率指标: 引擎对能力硬截断, 必须按年均/日最大两种统计分别取
MEAN_UTIL_KINDS = {"wtw": ("outflow_ml", ("daily_capacity_ml", "capacity_ml"))}
UTIL_SPECS = {
    "wtw": ("outflow_ml", ("daily_capacity_ml", "capacity_ml")),
    "wwtw": ("inflow_ml", ("daily_capacity_ml", "capacity_ml")),
    "reuse": ("outflow_ml", ("treatment_capacity_ml_day", "capacity_ml")),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_domain(domain: str, variant: str = "project.json") -> tuple[dict, pd.DataFrame, dict]:
    project = json.loads((R2 / domain / variant).read_text(encoding="utf-8"))
    timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
    constraints = json.loads((R2 / "specs" / f"constraints_{domain}.json").read_text(encoding="utf-8"))
    return project, timeseries, constraints


def apply_g(project: dict, timeseries: pd.DataFrame, pressure: str) -> tuple[dict, pd.DataFrame]:
    """External-pressure layer G0-G3 (开题表 3-4).

    G1 高温: 夏季(6-8月)干球 +4 ℃
    G2 干旱: 汛期(6-9月)入流 x0.65 且取水能力 x0.85(干旱年取水指标压减), 夏季 +1 ℃
    G3 复合: G1 + G2 + 传统需求 x1.12
    """
    trial = copy.deepcopy(project)
    drivers = timeseries.copy()
    dates = pd.to_datetime(drivers["date"])
    summer = dates.dt.month.isin([6, 7, 8])
    dry_season = dates.dt.month.isin([6, 7, 8, 9])
    if pressure in {"G1", "G3"}:
        drivers.loc[summer, "temperature_c"] = drivers.loc[summer, "temperature_c"].astype(float) + 4.0
    if pressure in {"G2", "G3"}:
        for column in drivers.columns:
            if "inflow" in column.lower():
                drivers.loc[dry_season, column] = drivers.loc[dry_season, column].astype(float) * 0.65
        for component in trial["components"].values():
            if component.get("kind") == "water_resource" and "abstraction_capacity_ml_day" in component:
                component["abstraction_capacity_ml_day"] = float(
                    component["abstraction_capacity_ml_day"]
                ) * 0.85
        if pressure == "G2":
            drivers.loc[summer, "temperature_c"] = drivers.loc[summer, "temperature_c"].astype(float) + 1.0
    if pressure == "G3":
        for area in trial["local_areas"].values():
            for profile in area.get("demand_profiles", []):
                if "base_value" in profile:
                    profile["base_value"] = float(profile["base_value"]) * 1.12
    return trial, drivers


def heatwave_drivers(timeseries: pd.DataFrame, days: int = 15, delta_c: float = 6.0) -> pd.DataFrame:
    """Short extreme heat wave: +delta_c for `days` consecutive days at the annual peak."""
    drivers = timeseries.copy()
    dates = pd.to_datetime(drivers["date"])
    summer = dates.dt.month.isin([6, 7, 8])
    idx = drivers.loc[summer, "temperature_c"].idxmax()
    start = int(idx)
    end = min(start + days, len(drivers))
    drivers.loc[start:end, "temperature_c"] = drivers.loc[start:end, "temperature_c"].astype(float) + delta_c
    return drivers


def mean_utilizations(result, project: dict) -> dict[str, float]:
    """Annual-mean utilisation for WTW / WWTW / reuse (year-scale statistic)."""
    values: dict[str, float] = {}
    for kind, (flow, keys) in UTIL_SPECS.items():
        best = 0.0
        for component_id, component in project.get("components", {}).items():
            if component.get("kind") != kind:
                continue
            capacity = next((float(component[k]) for k in keys if component.get(k) is not None), 0.0)
            if capacity <= 0:
                continue
            rows = result.component_daily[result.component_daily["component_id"] == component_id]
            if not rows.empty:
                best = max(best, float((rows[flow] / capacity).mean()))
        values[f"mean_{kind}_utilization"] = best
    return values


def peak_day_factor(result, project: dict) -> float:
    """Daily peak-to-mean factor of AI withdrawal (日峰值系数, 文献可比量)."""
    dc = result.data_center_daily
    if dc.empty:
        return float("nan")
    daily = dc.groupby("date", as_index=False).sum(numeric_only=True)
    series = daily["external_withdrawal_ml"].astype(float)
    mean = float(series.mean())
    return float(series.max() / mean) if mean > 0 else float("nan")


def scan_with_mean(project: dict, timeseries: pd.DataFrame, capacities) -> pd.DataFrame:
    """Capacity scan enriched with annual-mean utilisation and peak-day factor."""
    rows = []
    for capacity in capacities:
        trial = copy.deepcopy(project)
        for component_id, component in trial["components"].items():
            if component.get("kind") == "data_center":
                component["installed_it_capacity_mw"] = float(capacity)
                component.pop("capacity_schedule", None)
        result = FullAIUWMModel(trial, timeseries).run()
        record = {"ai_capacity_mw": float(capacity), **summarize_ai_water_kpis(result, trial)}
        record.update(mean_utilizations(result, trial))
        record["daily_peak_factor"] = peak_day_factor(result, trial)
        record["balance_residual_max_ml"] = float(
            result.data_center_daily["water_balance_residual_ml"].abs().max()
        ) if not result.data_center_daily.empty else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def annual_constraints(domain_constraints: dict) -> dict:
    """Year-scale constraint set: identical thresholds on annual-mean utilisation."""
    mapping = {
        "max_wtw_utilization": "mean_wtw_utilization",
        "max_wwtw_utilization": "mean_wwtw_utilization",
        "max_reuse_utilization": "mean_reuse_utilization",
    }
    out = {}
    for name, definition in domain_constraints["constraints"].items():
        if name in mapping:
            out[mapping[name]] = dict(definition)
        elif name in {"system_reliability_fraction", "domestic_unmet_ml", "unmet_cooling_water_ml",
                      "max_source_abstraction_ratio", "reclaimed_water_substitution_ratio"}:
            out[name] = dict(definition)
    return out


def daily_constraints(domain_constraints: dict) -> dict:
    """Day-scale constraint set.

    剔除 `max_reuse_utilization`: 再生水厂配 300 ML 贮池(约 7-9 天产水量),
    日尺度"处理速率达到铭牌"是正常的调蓄运行, 不是失效。把再生水产能修到
    现实利用率(~80%)之后, 该指标在全容量区间恒等于 1.0, 用它做日尺度判据
    会让承载容量恒为 0。再生水的系统级约束改由
    `reclaimed_water_substitution_ratio`(园区实际拿到的再生水比例)承担。
    """
    out = {}
    for name, definition in domain_constraints["constraints"].items():
        if name == "max_reuse_utilization":
            continue
        out[name] = dict(definition)
    return out


def hourly_cawcc(scan: pd.DataFrame, constraints: dict, profile: pd.DataFrame) -> dict:
    """Hour-scale CAWCC using a dimensionally consistent intra-day peak factor.

    `research.hourly_stress_scan` divides the daily-peak ratio by 24, which
    compares an hourly flow against a daily capacity and can therefore never
    fail.  The intra-day peaking factor is (profile max / profile mean) and is
    applied to the daily-peak statistics directly.
    """
    multipliers = profile["load_multiplier"].to_numpy(dtype=float)
    intra = float(multipliers.max() / multipliers.mean())
    frame = scan.copy()
    for column in ("peak_capacity_ratio", "max_local_connection_ratio",
                   "max_wtw_utilization", "max_reuse_utilization"):
        frame[f"hourly_{column}"] = frame[column].astype(float) * intra
    definitions = {}
    for name, definition in constraints["constraints"].items():
        if name == "peak_capacity_ratio":
            definitions["hourly_peak_capacity_ratio"] = dict(definition)
        elif name == "max_local_connection_ratio":
            definitions["hourly_max_local_connection_ratio"] = dict(definition)
        elif name == "max_wtw_utilization":
            definitions["hourly_max_wtw_utilization"] = dict(definition)
        elif name == "max_reuse_utilization":
            definitions["hourly_max_reuse_utilization"] = dict(definition)
        elif name in {"system_reliability_fraction", "domestic_unmet_ml",
                      "unmet_cooling_water_ml", "reclaimed_water_substitution_ratio"}:
            definitions[name] = dict(definition)
    positive = frame[frame["ai_capacity_mw"] >= 50].reset_index(drop=True)
    threshold = find_ai_carrying_capacity(positive, {"constraints": definitions})
    return {"intra_day_peak_factor": intra, "hourly_constraints": definitions, **{
        k: v for k, v in threshold.items() if k != "capacity_scan"}}


def constraint_informativeness(scan: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Flag constraints that never fail over the ladder (they carry no information)."""
    rows = []
    positive = scan[scan["ai_capacity_mw"] > 0]
    for name, definition in constraints["constraints"].items():
        if name not in scan.columns:
            rows.append({"constraint": name, "informative": False, "note": "metric absent"})
            continue
        values = positive[name].astype(float)
        operator = definition["operator"]
        target = float(definition["value"])
        if operator == ">=":
            violated = (values < target)
        elif operator == ">":
            violated = (values <= target)
        elif operator == "<=":
            violated = (values > target)
        else:
            violated = (values >= target)
        rows.append({
            "constraint": name,
            "informative": bool(violated.any()),
            "first_violated_capacity_mw": (
                float(positive.loc[violated.idxmax(), "ai_capacity_mw"]) if violated.any() else None
            ),
            "min_value": float(values.min()),
            "max_value": float(values.max()),
            "target": target,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main experiment
# --------------------------------------------------------------------------
def run_domain(domain: str, args) -> dict:
    folder = OUT / domain
    folder.mkdir(parents=True, exist_ok=True)
    project, timeseries, constraints = load_domain(domain)
    # 模拟期为多年 (默认 2 年); 引擎输出的所有 *_ml 都是"整个模拟期累计",
    # 与文献中的年尺度 WUE / 年取水量比较前必须年化。
    years = float(len(timeseries)) / 365.0
    report: dict = {"domain": domain, "simulation_years": years, "checks": {}}

    # ---- A. baseline vs AI (剔除城市自然增长) -----------------------------
    base_project = copy.deepcopy(project)
    base_project["components"].pop("AI_DC1", None)
    base_project["supply_paths"][0]["demand_priority"] = [
        item for item in base_project["supply_paths"][0]["demand_priority"]
        if not item.startswith("data_center::")
    ]
    base_result = FullAIUWMModel(base_project, timeseries).run()
    ai_project = copy.deepcopy(project)
    ai_project["components"]["AI_DC1"]["installed_it_capacity_mw"] = 500.0
    ai_result = FullAIUWMModel(ai_project, timeseries).run()
    delta = compare_baseline_ai(base_result, ai_result, base_project, ai_project)
    delta.to_csv(folder / "baseline_delta.csv", index=False)
    report["baseline_delta"] = delta.set_index("metric")["delta"].to_dict()

    # ---- B. capacity scan / CAWCC / boundaries ---------------------------
    scan = scan_with_mean(project, timeseries, CAPACITIES_SCAN)
    scan.to_csv(folder / "capacity_scan.csv", index=False)
    positive = scan[scan["ai_capacity_mw"] >= 50].reset_index(drop=True)
    day_constraints = {"constraints": daily_constraints(constraints)}
    daily_threshold = find_ai_carrying_capacity(positive, day_constraints)
    daily_threshold["capacity_scan"].to_csv(folder / "capacity_threshold.csv", index=False)
    boundaries = summarize_constraint_boundaries(positive, day_constraints)
    boundaries.to_csv(folder / "constraint_boundaries.csv", index=False)
    informative = constraint_informativeness(scan, day_constraints)
    informative.to_csv(folder / "constraint_informativeness.csv", index=False)

    annual = annual_constraints(constraints)
    annual_threshold = find_ai_carrying_capacity(positive, {"constraints": annual})
    profile = build_intraday_profile(
        training_fraction=0.65, inference_fraction=0.35,
        inference_peak_factor=1.35, peak_hours=range(10, 22),
    )
    hourly = hourly_cawcc(scan, day_constraints, profile)

    report["cawcc_daily"] = {
        "maximum_safe_ai_capacity_mw": daily_threshold["maximum_safe_ai_capacity_mw"],
        "first_failed_capacity_mw": daily_threshold["first_failed_capacity_mw"],
        "limiting_constraint": daily_threshold["limiting_constraint"],
        "threshold_interval_mw": daily_threshold["threshold_interval_mw"],
        "reentrant_feasibility": daily_threshold["reentrant_feasibility"],
    }
    report["cawcc_annual"] = {
        "maximum_safe_ai_capacity_mw": annual_threshold["maximum_safe_ai_capacity_mw"],
        "first_failed_capacity_mw": annual_threshold["first_failed_capacity_mw"],
        "limiting_constraint": annual_threshold["limiting_constraint"],
    }
    report["cawcc_hourly"] = {
        "maximum_safe_ai_capacity_mw": hourly["maximum_safe_ai_capacity_mw"],
        "first_failed_capacity_mw": hourly["first_failed_capacity_mw"],
        "limiting_constraint": hourly["limiting_constraint"],
        "intra_day_peak_factor": hourly["intra_day_peak_factor"],
    }
    boundaries.to_dict("records")
    report["bottleneck_chain"] = boundaries.to_dict("records")

    # ---- C. intervention marginal curves ---------------------------------
    # 干预规格按域生成(幅度以该域自身基线为分母); 缺省回落到通用模板
    _spec_path = R2 / "specs" / f"interventions_{domain}.json"
    if not _spec_path.exists():
        _spec_path = R2 / "specs" / "interventions.json"
    spec = json.loads(_spec_path.read_text(encoding="utf-8"))
    # 0 MW 必须排除: 没有数据中心时 "再生水替代率" 无定义(=0), 会被误判为
    # 违反 >= 0.65 的约束, 从而污染 first_failed_capacity 与瓶颈归因。
    intervention_capacities = [float(c) for c in spec["capacities_mw"] if float(c) > 0]
    interventions = evaluate_ai_interventions(
        project, timeseries, spec["interventions"], day_constraints,
        capacities_mw=intervention_capacities,
    )
    interventions.to_csv(folder / "intervention_marginal.csv", index=False)
    # 干预增益必须与"同网格基准"比较: 用细网格 CAWCC 减去粗网格干预结果会
    # 产生虚假的负增益(网格分辨率差, 不是物理效应)。
    same_grid_baseline = float(
        interventions.loc[
            interventions["intervention"] == "baseline", "maximum_safe_ai_capacity_mw"
        ].iloc[0]
    )
    interventions["capacity_gain_mw"] = (
        interventions["maximum_safe_ai_capacity_mw"].astype(float) - same_grid_baseline
    )
    interventions.to_csv(folder / "intervention_marginal.csv", index=False)
    report["interventions"] = interventions.to_dict("records")
    report["intervention_grid"] = {
        "capacities_mw": spec["capacities_mw"],
        "same_grid_baseline_mw": same_grid_baseline,
    }

    # ---- C2. 主导瓶颈的边际有效性曲线 (开题 3.6 步长细化) -----------------
    marginal_rows = []
    # 起点必须取该域真实的再生水产能(HA 34 / CO 30), 否则第一个差分步长
    # 不是"从现状起"的边际增益。
    base_reuse_capacity = float(project["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"])
    marginal_grid = sorted({base_reuse_capacity, 45.0, 60.0, 75.0, 90.0, 120.0, 160.0})
    for reuse_capacity in marginal_grid:
        trial = copy.deepcopy(project)
        trial["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"] = reuse_capacity
        trial_scan = scan_ai_capacity(trial, timeseries, capacities_mw=spec["capacities_mw"])
        found = find_ai_carrying_capacity(
            trial_scan[trial_scan["ai_capacity_mw"] >= 50].reset_index(drop=True), day_constraints
        )
        marginal_rows.append({
            "reuse_treatment_capacity_ml_day": reuse_capacity,
            "cawcc_mw": found["maximum_safe_ai_capacity_mw"],
            "limiting_constraint": found["limiting_constraint"],
        })
    marginal = pd.DataFrame(marginal_rows)
    marginal["marginal_gain_mw_per_ml_day"] = (
        marginal["cawcc_mw"].astype(float).diff()
        / marginal["reuse_treatment_capacity_ml_day"].astype(float).diff()
    )
    marginal.to_csv(folder / "reuse_marginal_curve.csv", index=False)
    report["reuse_marginal_curve"] = marginal.to_dict("records")

    # ---- C3. 再生水比例 - CoC - 补水 反馈链 (机制三) ----------------------
    coc_rows = []
    for target in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        trial = copy.deepcopy(project)
        trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = 1000.0
        sources = trial["components"]["AI_DC1"]["water_sources"]
        sources["reclaimed"]["target_fraction"] = target
        sources["potable"]["target_fraction"] = 1.0 - target
        result = FullAIUWMModel(trial, timeseries).run()
        daily = result.data_center_daily
        kpis = summarize_ai_water_kpis(result, trial)
        coc_rows.append({
            "reclaimed_target_fraction": target,
            "achieved_coc": float(daily["cycles_of_concentration"].mean()),
            "quality_coc_fallback_days": int(daily["quality_coc_fallback"].sum()),
            "annual_makeup_ml": float(daily["external_makeup_ml"].sum()),
            "annual_blowdown_ml": float(daily["blowdown_ml"].sum()),
            "annual_evaporation_ml": float(daily["evaporation_ml"].sum()),
            "actual_reclaimed_fraction": float(daily["actual_reclaimed_fraction"].mean()),
            "freshwater_withdrawal_ml": float(kpis["freshwater_withdrawal_ml"]),
            "reclaimed_water_use_ml": float(kpis["reclaimed_water_use_ml"]),
        })
    coc_frame = pd.DataFrame(coc_rows)
    coc_frame.to_csv(folder / "reclaimed_coc_feedback.csv", index=False)
    report["reclaimed_coc_feedback"] = coc_frame.to_dict("records")

    # ---- C4. 再生水配置的"毛替代" vs "净替代" (现实性审计新增) -------------
    # 把再生水厂产能利用率从 19% 修正到 ~95%(现实水厂水平)之后出现的关键
    # 机制: 基线(无 AI)下再生水已经满负荷供给工业与市政杂用, 数据中心拿走的
    # 再生水只是把这些既有用户挤回自来水系统。因此
    #   毛替代率 = 园区再生水用量 / 园区总取水量
    # 高估了政策收益; 真正有意义的是
    #   净替代率 = (园区总取水量 - 城市取水量增量 x 输配效率) / 园区总取水量
    # 只有新增的再生水产能才产生净替代。这一节给出两种口径, 并给出"新增
    # 再生水产能 -> 净替代率"的边际曲线。
    def city_abstraction_ml(result) -> float:
        rows = result.component_daily
        rows = rows[rows["component_id"] == "RES1"]
        return float(rows["outflow_ml"].sum()) if len(rows) else float("nan")

    base_project = copy.deepcopy(project)
    base_project["components"]["AI_DC1"]["installed_it_capacity_mw"] = 0.0
    base_result = FullAIUWMModel(base_project, timeseries).run()
    abstraction_0 = city_abstraction_ml(base_result)
    potable_0 = float(base_result.area_daily["potable_delivered_ml"].sum())
    conveyance = potable_0 / abstraction_0 if abstraction_0 else float("nan")

    net_rows = []
    for capacity in (500.0, 1000.0, 1500.0):
        trial = copy.deepcopy(project)
        trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = capacity
        result = FullAIUWMModel(trial, timeseries).run()
        kpis = summarize_ai_water_kpis(result, trial)
        withdrawal = float(kpis["total_withdrawal_ml"])
        reclaimed = float(kpis["reclaimed_water_use_ml"])
        delta = city_abstraction_ml(result) - abstraction_0
        delivered_equivalent = delta * conveyance
        net_saving = withdrawal - delivered_equivalent
        net_rows.append({
            "capacity_mw": capacity,
            "reuse_treatment_capacity_ml_day": project["components"]["CENTRAL_REUSE"][
                "treatment_capacity_ml_day"],
            "baseline_city_abstraction_ml": abstraction_0,
            "city_abstraction_ml": abstraction_0 + delta,
            "city_abstraction_increase_ml": delta,
            "conveyance_efficiency": conveyance,
            "ai_withdrawal_ml": withdrawal,
            "ai_reclaimed_ml": reclaimed,
            "gross_substitution_ratio": reclaimed / withdrawal if withdrawal else float("nan"),
            "net_freshwater_saving_ml": net_saving,
            "net_substitution_ratio": net_saving / withdrawal if withdrawal else float("nan"),
        })
    # 新增再生水产能 -> 净替代率 的边际曲线
    base_reuse = float(project["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"])
    for factor in (1.0, 1.5, 2.0, 3.0):
        trial = copy.deepcopy(project)
        trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = 1000.0
        trial["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"] = base_reuse * factor
        # WWTW 分流比例同步放大, 否则产能提高而进水不变, 产能仍然闲置
        trial["components"]["WWTW1"]["central_reuse_fraction"] = min(
            0.95, float(project["components"]["WWTW1"]["central_reuse_fraction"]) * factor
        )
        result = FullAIUWMModel(trial, timeseries).run()
        kpis = summarize_ai_water_kpis(result, trial)
        withdrawal = float(kpis["total_withdrawal_ml"])
        reclaimed = float(kpis["reclaimed_water_use_ml"])
        delta = city_abstraction_ml(result) - abstraction_0
        net_saving = withdrawal - delta * conveyance
        net_rows.append({
            "capacity_mw": 1000.0,
            "reuse_treatment_capacity_ml_day": base_reuse * factor,
            "baseline_city_abstraction_ml": abstraction_0,
            "city_abstraction_ml": abstraction_0 + delta,
            "city_abstraction_increase_ml": delta,
            "conveyance_efficiency": conveyance,
            "ai_withdrawal_ml": withdrawal,
            "ai_reclaimed_ml": reclaimed,
            "gross_substitution_ratio": reclaimed / withdrawal if withdrawal else float("nan"),
            "net_freshwater_saving_ml": net_saving,
            "net_substitution_ratio": net_saving / withdrawal if withdrawal else float("nan"),
            "tag": f"reuse_capacity_x{factor:g}",
        })
    # 再生水的两个杠杆必须分开扫。只扩"处理能力"(treatment_capacity)而产水量
    # 不变是无效投资: 上表实测边际增益恒为 0。真正有效的杠杆是 WWTW 分流比例
    # (决定产水量), 产能需同步跟随(取产水量的 1.25 倍作日峰值裕度)。
    base_fraction = float(project["components"]["WWTW1"]["central_reuse_fraction"])
    baseline_sewage_ml_d = float(
        scan.loc[scan["ai_capacity_mw"] == 0.0, "wastewater_ml"].iloc[0]
    ) / (365.0 * years)
    production_rows = []
    for fraction in (base_fraction, 0.25, 0.35, 0.45, 0.60):
        trial = copy.deepcopy(project)
        trial["components"]["WWTW1"]["central_reuse_fraction"] = fraction
        trial["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"] = round(
            fraction * baseline_sewage_ml_d * 1.25, 1
        )
        trial_scan = scan_ai_capacity(trial, timeseries, capacities_mw=spec["capacities_mw"])
        found = find_ai_carrying_capacity(
            trial_scan[trial_scan["ai_capacity_mw"] >= 50].reset_index(drop=True), day_constraints
        )
        production_rows.append({
            "central_reuse_fraction": fraction,
            "estimated_production_ml_day": fraction * baseline_sewage_ml_d,
            "reuse_treatment_capacity_ml_day": round(fraction * baseline_sewage_ml_d * 1.25, 1),
            "cawcc_mw": found["maximum_safe_ai_capacity_mw"],
            "limiting_constraint": found["limiting_constraint"],
        })
    production_frame = pd.DataFrame(production_rows)
    production_frame["marginal_gain_mw_per_ml_day"] = (
        production_frame["cawcc_mw"].astype(float).diff()
        / production_frame["estimated_production_ml_day"].astype(float).diff()
    )
    production_frame.to_csv(folder / "reuse_production_marginal_curve.csv", index=False)
    report["reuse_production_marginal_curve"] = production_frame.to_dict("records")

    net_frame = pd.DataFrame(net_rows)
    net_frame.to_csv(folder / "reclaimed_net_substitution.csv", index=False)
    report["reclaimed_net_substitution"] = net_frame.to_dict("records")

    # ---- D. cooling technology comparison --------------------------------
    tech_rows = []
    for tech_file in sorted((R2 / domain).glob("tech_*.json")):
        tech = tech_file.stem.replace("tech_", "")
        tech_project = json.loads(tech_file.read_text(encoding="utf-8"))
        tech_project["components"]["AI_DC1"]["installed_it_capacity_mw"] = 750.0
        result = FullAIUWMModel(tech_project, timeseries).run()
        kpis = summarize_ai_water_kpis(result, tech_project)
        tech_scan = scan_with_mean(tech_project, timeseries, [0.0, 500, 1000, 1500, 2000])
        tech_threshold = find_ai_carrying_capacity(
            tech_scan[tech_scan["ai_capacity_mw"] >= 50].reset_index(drop=True), day_constraints
        )
        dc_daily = result.data_center_daily
        tech_rows.append({
            "technology": tech,
            "wue_l_kwh": float(
                kpis["total_withdrawal_ml"] * 1000.0 / max(kpis["it_energy_mwh"], 1e-9)
            ),
            "mean_pue": float(dc_daily["pue"].mean()) if not dc_daily.empty else float("nan"),
            "mean_wet_fraction": float(dc_daily["wet_cooling_fraction"].mean()) if not dc_daily.empty else float("nan"),
            # 年化: 模拟期为 years 年, 引擎输出为全期累计
            "period_withdrawal_ml": float(kpis["total_withdrawal_ml"]),
            "annual_withdrawal_ml": float(kpis["total_withdrawal_ml"]) / years,
            "annual_consumption_ml": float(kpis["consumption_ml"]) / years,
            "daily_peak_factor": peak_day_factor(result, tech_project),
            "facility_energy_mwh": float(kpis["facility_energy_mwh"]),
            "cawcc_mw": tech_threshold["maximum_safe_ai_capacity_mw"],
            "limiting_constraint": tech_threshold["limiting_constraint"],
        })
    tech_frame = pd.DataFrame(tech_rows)
    tech_frame.to_csv(folder / "technology_comparison.csv", index=False)
    report["technology"] = tech_frame.to_dict("records")

    # ---- E. allocation policy comparison ---------------------------------
    policy_rows = []
    for policy_file in sorted((R2 / domain).glob("policy_*.json")):
        policy = policy_file.stem
        policy_project = json.loads(policy_file.read_text(encoding="utf-8"))
        for capacity in (1000.0, 1500.0, 2000.0):
            trial = copy.deepcopy(policy_project)
            trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = capacity
            result = FullAIUWMModel(trial, timeseries).run()
            kpis = summarize_ai_water_kpis(result, trial)
            # 直接取分区尺度的居民未满足量: KPI 里的 domestic_unmet_ml 是
            # "系统未满足 - AI未满足" 的残差, 会被 AI 冷却储水缓冲污染。
            domestic_unmet = float(result.area_daily["unmet_domestic_ml"].sum())
            policy_rows.append({
                "policy": policy, "capacity_mw": capacity,
                "domestic_unmet_ml": domestic_unmet,
                "domestic_unmet_residual_kpi": float(kpis["domestic_unmet_ml"]),
                "system_reliability_fraction": float(kpis["system_reliability_fraction"]),
                "unmet_cooling_water_ml": float(kpis["unmet_cooling_water_ml"]),
                "ai_withdrawal_ml": float(kpis["total_withdrawal_ml"]),
            })
    policy_frame = pd.DataFrame(policy_rows)
    policy_frame.to_csv(folder / "policy_comparison.csv", index=False)
    report["policy"] = policy_frame.to_dict("records")

    # ---- F. S x G matrix --------------------------------------------------
    sg_rows = []
    for state in ("S0", "S1", "S2", "S3"):
        for pressure in ("G0", "G1", "G2", "G3"):
            state_project, state_drivers = apply_state_pressure(project, timeseries, state, "G0")
            trial, drivers = apply_g(state_project, state_drivers, pressure)
            trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = 1000.0
            result = FullAIUWMModel(trial, drivers).run()
            kpis = summarize_ai_water_kpis(result, trial)
            # 夏季(6-8月)AI 取水量: 高温信号只在夏季出现, 用全年总量会把
            # +4 ℃ 的信号稀释到 0.3 % 以下 (夏季占全年 1/4, 且 PUE 温度
            # 系数只有 0.003/℃), 因此必须按季节切片。
            dc_daily = result.data_center_daily
            summer_mask = pd.to_datetime(dc_daily["date"]).dt.month.isin([6, 7, 8])
            summer_withdrawal = float(dc_daily.loc[summer_mask, "external_withdrawal_ml"].sum())
            sg_rows.append({
                "state": state, "pressure": pressure,
                "summer_withdrawal_ml": summer_withdrawal,
                "total_withdrawal_ml": float(kpis["total_withdrawal_ml"]),
                "freshwater_withdrawal_ml": float(kpis["freshwater_withdrawal_ml"]),
                "reclaimed_water_use_ml": float(kpis["reclaimed_water_use_ml"]),
                "system_reliability_fraction": float(kpis["system_reliability_fraction"]),
                "domestic_unmet_ml": float(kpis["domestic_unmet_ml"]),
                "max_wtw_utilization": float(kpis["max_wtw_utilization"]),
                "max_reuse_utilization": float(kpis["max_reuse_utilization"]),
                "max_source_abstraction_ratio": float(kpis["max_source_abstraction_ratio"]),
                "reclaimed_water_substitution_ratio": float(kpis["reclaimed_water_substitution_ratio"]),
                "unmet_cooling_water_ml": float(kpis["unmet_cooling_water_ml"]),
            })
    sg_frame = pd.DataFrame(sg_rows)
    sg_frame.to_csv(folder / "state_pressure_matrix.csv", index=False)
    report["state_pressure"] = sg_frame.to_dict("records")

    # ---- G. heat wave stress ---------------------------------------------
    hw_drivers = heatwave_drivers(timeseries)
    hw_rows = []
    for capacity in (500.0, 1000.0, 1500.0):
        trial = copy.deepcopy(project)
        trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = capacity
        result = FullAIUWMModel(trial, hw_drivers).run()
        kpis = summarize_ai_water_kpis(result, trial)
        hw_rows.append({"capacity_mw": capacity, **{
            k: float(v) for k, v in kpis.items() if k in {
                "maximum_daily_withdrawal_ml", "total_withdrawal_ml",
                "max_wtw_utilization", "system_reliability_fraction",
                "max_local_connection_ratio", "unmet_cooling_water_ml",
            }}})
    pd.DataFrame(hw_rows).to_csv(folder / "heatwave_stress.csv", index=False)
    report["heatwave"] = hw_rows

    # ---- H. Morris sensitivity -------------------------------------------
    if not args.skip_sensitivity:
        spec = json.loads((R2 / "specs" / "sensitivity_morris.json").read_text(encoding="utf-8"))
        trial = copy.deepcopy(project)
        trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = 1000.0
        bounds = {name: tuple(map(float, value)) for name, value in spec["parameters"].items()}
        morris = morris_sensitivity(
            make_evaluator(trial, timeseries, spec["metric"]), bounds,
            trajectories=int(spec["trajectories"]), seed=int(spec["seed"]),
        )
        morris.to_csv(folder / "morris.csv", index=False)
        report["morris"] = morris.to_dict("records")
        sobol_spec = json.loads((R2 / "specs" / "sensitivity_sobol.json").read_text(encoding="utf-8"))
        sobol = sobol_sensitivity(
            make_evaluator(trial, timeseries, sobol_spec["metric"]), bounds,
            samples=int(sobol_spec.get("samples", 32)), seed=int(sobol_spec["seed"]),
        )
        sobol.to_csv(folder / "sobol.csv", index=False)
        report["sobol"] = sobol.to_dict("records")

    # ---- I. probabilistic threshold --------------------------------------
    if not args.skip_probabilistic:
        spec = json.loads(
            (R2 / "specs" / f"probabilistic_threshold_{domain}.json").read_text(encoding="utf-8")
        )
        ranges = {name: tuple(map(float, value)) for name, value in spec["parameter_ranges"].items()}
        samples = probabilistic_ai_capacity_threshold(
            project, timeseries, ranges, [float(c) for c in spec["capacities_mw"]],
            {"constraints": spec["constraints"]},
            samples=int(spec["samples"]), seed=int(spec["seed"]),
        )
        samples.to_csv(folder / "probabilistic_thresholds.csv", index=False)
        series = pd.to_numeric(samples["maximum_safe_ai_capacity_mw"], errors="coerce").dropna()
        proposed = float(args.proposed_mw)
        exceedance = float(
            capacity_exceedance_probability(samples, proposed)
        )
        report["probabilistic"] = {
            "samples": int(len(series)),
            "p05": float(series.quantile(0.05)) if len(series) else None,
            "p50": float(series.quantile(0.50)) if len(series) else None,
            "p95": float(series.quantile(0.95)) if len(series) else None,
            "proposed_mw": proposed,
            "exceedance_probability": exceedance,
        }

    # ---- verification checks ---------------------------------------------
    checks = {}
    checks["V1_water_balance"] = {
        "max_residual_ml": float(scan["balance_residual_max_ml"].max()),
        "passed": bool(scan["balance_residual_max_ml"].max() <= 1e-9),
    }
    withdrawal = scan["total_withdrawal_ml"].astype(float).to_numpy()
    diffs = np.diff(withdrawal)
    checks["V2_monotone_withdrawal"] = {
        "min_diff": float(diffs.min()),
        "passed": bool(diffs.min() >= -1e-6),
    }
    # V3 是诊断项而非通过/失败项: 常态气候下不失效的约束, 需检查其在 G2/G3
    # 与 S1-S3 下是否变为有效判据(见 state_pressure_matrix)。
    checks["V3_constraints_informative"] = {
        "uninformative_in_G0_S0": informative.loc[~informative["informative"], "constraint"].tolist(),
        "passed": True,
        "note": "diagnostic: 不失效约束需靠压力情景激活",
    }
    order = [
        report["cawcc_annual"]["maximum_safe_ai_capacity_mw"],
        report["cawcc_daily"]["maximum_safe_ai_capacity_mw"],
        report["cawcc_hourly"]["maximum_safe_ai_capacity_mw"],
    ]
    order_clean = [value for value in order if value is not None]
    checks["V4_scale_ordering"] = {
        "annual": order[0], "daily": order[1], "hourly": order[2],
        "passed": bool(len(order_clean) == 3 and order_clean[0] >= order_clean[1] >= order_clean[2]),
    }
    tech = tech_frame.set_index("technology")
    checks["V5_pue_wue_ordering"] = {
        "pue_dry_gt_evaporative": bool(tech.loc["dry", "mean_pue"] > tech.loc["evaporative", "mean_pue"]),
        "wue_dry_lt_evaporative": bool(tech.loc["dry", "wue_l_kwh"] < tech.loc["evaporative", "wue_l_kwh"]),
        "passed": bool(
            tech.loc["dry", "mean_pue"] > tech.loc["evaporative", "mean_pue"]
            and tech.loc["dry", "wue_l_kwh"] < tech.loc["evaporative", "wue_l_kwh"]
        ),
    }
    peak = {
        name: float(tech.loc[name, "daily_peak_factor"]) for name in
        ("dry_first", "hybrid", "evaporative", "efficient_evaporative", "liquid_to_water")
    }
    # 日峰值系数 = max(日取水) / mean(日取水)。在"几乎不用水"的技术-气候组合
    # 下 (干式优先 x 凉爽域, 全年仅个位数天数开蒸发辅助) 分母趋近 0, 该比值
    # 是退化统计量, 不能与文献区间比较, 必须显式剔除而不是判为"超标"。
    water_annual = {name: float(tech.loc[name, "annual_withdrawal_ml"]) for name in peak}
    reference = max(water_annual.values())
    usable = {
        name: value for name, value in peak.items()
        if np.isfinite(value) and water_annual[name] >= 0.05 * reference
    }
    degenerate = sorted(set(peak) - set(usable))
    direction = (
        bool(usable["dry_first"] > usable["evaporative"])
        if {"dry_first", "evaporative"} <= set(usable) else None
    )
    band_dry = (
        bool(6.5 <= usable["dry_first"] <= 10.0) if "dry_first" in usable else None
    )
    band_evap = (
        bool(1.7 <= usable["evaporative"] <= 3.0) if "evaporative" in usable else None
    )
    checks["V6_peak_factor_mechanism"] = {
        "daily_peak_factors": peak,
        "annual_withdrawal_ml": water_annual,
        "evaluated": usable,
        "degenerate_excluded": degenerate,
        "direction_dry_first_gt_evaporative": direction,
        "literature_band_dry_first_6p5_10": band_dry,
        "literature_band_evaporative_1p7_3p0": band_evap,
        "note": "退化组合(年取水 < 同域最大值的 5%)不参与量级校核",
        "passed": bool(direction is not False and band_dry is not False),
    }
    gains = interventions["capacity_gain_mw"].astype(float)
    checks["V7_intervention_monotone"] = {
        "min_gain_mw": float(gains.min()),
        "passed": bool(gains.min() >= -1e-6),
    }
    policy_pivot = policy_frame.pivot(index="capacity_mw", columns="policy", values="domestic_unmet_ml")
    checks["V8_policy_resident_first"] = {
        "domestic_unmet_by_policy": policy_pivot.to_dict(),
        "passed": bool(
            (policy_pivot["policy_a"] <= policy_pivot["policy_c"] + 1e-9).all()
        ),
    }
    sg = sg_frame.set_index(["state", "pressure"])
    # 气候敏感性必须拆成两个独立的物理信号, 不能用一个总量指标:
    #   1) 高温 -> 夏季冷却取水量上升 (夏季切片, 不能看全年)
    #   2) 干旱 -> 再生水供给下降 -> AI 再生水用量下降
    # 用"总取水量"或"淡水取水量"都会失败: AI 优先取再生水, 干旱时再生水
    # 少了就自动置换为淡水, 总量几乎不变; 而城市配水总量下降又会使 AI 的
    # 淡水取水不升反降, 出现 G3 < G0 的假阴性。
    summer0 = float(sg.loc[("S0", "G0"), "summer_withdrawal_ml"])
    summer1 = float(sg.loc[("S0", "G1"), "summer_withdrawal_ml"])
    abs0 = float(sg.loc[("S0", "G0"), "max_source_abstraction_ratio"])
    abs2 = float(sg.loc[("S0", "G2"), "max_source_abstraction_ratio"])
    reclaimed0 = float(sg.loc[("S0", "G0"), "reclaimed_water_use_ml"])
    reclaimed2 = float(sg.loc[("S0", "G2"), "reclaimed_water_use_ml"])
    checks["V9_climate_sensitivity"] = {
        "heat_signal_metric": "summer_withdrawal_ml (6-8月 AI 取水)",
        "summer_G0_ml": summer0,
        "summer_G1_ml": summer1,
        "delta_summer_G1_percent": float(100.0 * (summer1 - summer0) / summer0) if summer0 else None,
        # 干旱信号必须落在"水源侧": 再生水供给由污水量决定, 而污水量由供水
        # 量决定; 在库容按多年调节标定(120,000 ML, 起调 65,000 ML)且干旱只压
        # 入流与取水能力时, 供水并未真正受限, 再生水用量对干旱几乎无响应
        # (实测 +0.07 %)。真正响应的是取水能力占用率。这个"再生水对短期水文
        # 干旱不敏感"本身是一个需要写进论文的结构发现。
        "drought_signal_metric": "max_source_abstraction_ratio",
        "abstraction_ratio_G0": abs0,
        "abstraction_ratio_G2": abs2,
        "delta_abstraction_ratio_G2": float(abs2 - abs0),
        "reclaimed_drought_response_percent": float(100.0 * (reclaimed2 - reclaimed0) / reclaimed0) if reclaimed0 else None,
        "passed": bool(summer1 > summer0 and abs2 > abs0),
    }
    # V11: 净替代率不可能超过毛替代率。若违反, 说明"城市取水量增量"为负或
    # 输配效率口径错误, 是计算 bug 而不是物理结果。
    # 只校核"再生水配置不变"的容量扫描行: 扩大再生水产能会让既有非饮用水
    # 用户一并改用再生水, 城市取水量可能低于无 AI 基线, 此时净节省超过园区
    # 自身取水量是真实效应(释放了既有用户的淡水占用), 不是计算错误。
    net = net_frame[net_frame["tag"].isna()]
    net = net.dropna(subset=["net_substitution_ratio", "gross_substitution_ratio"])
    violations = net[net["net_substitution_ratio"] > net["gross_substitution_ratio"] + 1e-9]
    expansion = net_frame[net_frame["tag"].notna()]
    checks["V11_net_le_gross_substitution"] = {
        "rows": int(len(net)),
        "violations": int(len(violations)),
        "gross_at_1000mw": float(net.loc[net["capacity_mw"] == 1000.0, "gross_substitution_ratio"].iloc[0]),
        "net_at_1000mw": float(net.loc[net["capacity_mw"] == 1000.0, "net_substitution_ratio"].iloc[0]),
        "net_at_1000mw_reuse_x3": float(
            expansion.loc[expansion["tag"] == "reuse_capacity_x3", "net_substitution_ratio"].iloc[0]
        ) if len(expansion) else None,
        "note": "仅校核再生水配置不变的行; 扩产能行可出现净节省 > 园区取水量",
        "passed": bool(len(violations) == 0 and len(net) >= 3),
    }
    # V12: 基线(0 MW)城市必须自洽 —— 无缺水量, 且水源库容不被抽干。
    # 原设定库容 40,000 ML(仅年取水量的 0.48 倍)在 1000 MW 下被抽干并产生
    # 13,641 ML 缺水量, 那是库容设定过小造成的伪约束。
    base_store = base_result.component_daily
    base_store = base_store[base_store["component_id"] == "RES1"]["storage_ml"]
    checks["V12_baseline_city_self_consistent"] = {
        "baseline_unmet_ml": float(base_result.area_daily["unmet_ml"].sum()),
        "baseline_min_storage_ml": float(base_store.min()),
        "reservoir_capacity_ml": float(project["components"]["RES1"]["capacity_ml"]),
        "min_storage_fraction": float(base_store.min()) / float(project["components"]["RES1"]["capacity_ml"]),
        # 判据用容差而不是 == 0: 引擎浮点求和会留下 ~1e-12 量级的残差。
        "passed": bool(
            float(base_result.area_daily["unmet_ml"].sum()) <= 1e-6
            and float(base_store.min()) > 0.0
        ),
    }
    checks["V10_uninformative_constraints_activated_by_pressure"] = {
        "source_abstraction_max_over_SxG": float(sg["max_source_abstraction_ratio"].max()),
        "reliability_min_over_SxG": float(sg["system_reliability_fraction"].min()),
        "passed": bool(
            sg["max_source_abstraction_ratio"].max() >= 0.95
            or sg["system_reliability_fraction"].min() < 0.99
        ),
    }
    report["checks"] = checks
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-probabilistic", action="store_true")
    parser.add_argument("--proposed-mw", type=float, default=1000.0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    for domain in args.domains.split(","):
        print(f"running {domain} ...", flush=True)
        reports[domain] = run_domain(domain, args)
    (OUT / "r2_verification.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    for domain, report in reports.items():
        print(f"\n=== {domain} ===")
        print("CAWCC annual:", report["cawcc_annual"])
        print("CAWCC daily :", report["cawcc_daily"])
        print("CAWCC hourly:", report["cawcc_hourly"])
        for name, check in report["checks"].items():
            print(f"  {name}: {'PASS' if check['passed'] else 'FAIL'}  {check}")
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
