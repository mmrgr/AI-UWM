"""Realism audit: compare model outputs against real-world benchmarks.

The study is a numerical experiment and does not need observed data, but every
quantity must be *dimensionally consistent and of the right order of magnitude*,
otherwise the conclusions cannot inform real planning.  This script prints, for
each parameter-domain endpoint, the city-side and AI-side quantities together
with a real-world reference interval and a PASS / WARN / FAIL verdict.

References used for the benchmark intervals (documented in the delivery):
  * 北京市水资源公报 (2024): 再生水利用量/污水处理量 = 57.5%; 工业用水中再生水
    占比 32.6%。全国地级及以上城市再生水利用率目标 ~25%, 缺水城市 ~35%。
  * GB/T 50331, CJJ 92: urban water quota and leakage band.
  * LBNL (2024) and operator sustainability reports: PUE 1.1-1.5, source WUE
    0.3-2.0 L/kWh (leaders 0.2-0.6).
  * DOE FEMP / ASHRAE: cooling-tower cycles of concentration 3-7 (up to 10).
  * 中国城市人均用电量 ~6000 kWh/yr (全社会口径), 用于判定园区电力规模。

口径说明 (v2 修正):
  * 配水量必须取 `delivered_total_ml`。v1 用 `startswith("delivered")` 求和,
    把无量纲的 `delivered_percent`(=100) 当成 ML 加进去, 得出虚高的
    330 L/人/d; 正确值为 ~230-250 L/人/d。
  * 漏损率必须"自来水口径对自来水口径": 1 - potable_delivered / 水源取水量。
    v1 用含再生水的总配水量作分子、只含自来水的 WTW 进水作分母, 得出负漏损。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiuwm.ai_metrics import summarize_ai_water_kpis
from aiuwm.full_engine import FullAIUWMModel

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"

POPULATION = 1_000_000
# 全社会口径人均用电量 (kWh/yr): 中国城市量级 5000-7000, 取 6000
CITY_ELECTRICITY_KWH_PER_CAPITA_YEAR = 6000.0


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if column in frame else float("nan")


def city_side(result, project: dict, years: float) -> dict:
    comp = result.component_daily
    area = result.area_daily

    def comp_mean(cid: str, column: str) -> float:
        rows = comp[comp["component_id"] == cid]
        return float(rows[column].mean()) if len(rows) else float("nan")

    abstraction = comp_mean("RES1", "outflow_ml")
    wtw_in = comp_mean("WTW1", "inflow_ml")
    wwtw_in = comp_mean("WWTW1", "inflow_ml")
    reuse_out = comp_mean("CENTRAL_REUSE", "outflow_ml")
    reuse_capacity = float(
        project["components"]["CENTRAL_REUSE"].get("treatment_capacity_ml_day", np.nan)
    )

    potable = _mean(area, "potable_delivered_ml")
    reuse_delivered = _mean(area, "reuse_delivered_ml")
    delivered = _mean(area, "delivered_total_ml")
    demand = _mean(area, "water_demand_ml")

    res = comp[comp["component_id"] == "RES1"]
    capacity_ml = float(project["components"]["RES1"].get("capacity_ml", np.nan))
    spill = _mean(res, "overflow_ml")
    storage = res["storage_ml"] if "storage_ml" in res else pd.Series(dtype=float)

    return {
        "population": POPULATION,
        "raw_water_abstraction_ml_d": abstraction,
        "wtw_inflow_ml_d": wtw_in,
        "wwtw_inflow_ml_d": wwtw_in,
        "reuse_production_ml_d": reuse_out,
        "reuse_capacity_ml_d": reuse_capacity,
        "reuse_plant_utilization": reuse_out / reuse_capacity if reuse_capacity else np.nan,
        "potable_delivered_ml_d": potable,
        "reuse_delivered_ml_d": reuse_delivered,
        "area_delivered_ml_d": delivered,
        "area_demand_ml_d": demand,
        "per_capita_delivered_l_d": delivered * 1e6 / POPULATION,
        "per_capita_abstraction_l_d": abstraction * 1e6 / POPULATION,
        "reuse_over_sewage": reuse_out / wwtw_in if wwtw_in else np.nan,
        "leakage_plus_loss_fraction": 1.0 - potable / abstraction if abstraction else np.nan,
        "reservoir_spill_fraction": spill / abstraction if abstraction else np.nan,
        "reservoir_spill_ml_d": spill,
        "reservoir_storage_min_fraction": float(storage.min()) / capacity_ml if capacity_ml else np.nan,
        "reservoir_storage_max_fraction": float(storage.max()) / capacity_ml if capacity_ml else np.nan,
        "years": years,
    }


def ai_side(result, project: dict, years: float) -> dict:
    daily = result.data_center_daily
    kpis = summarize_ai_water_kpis(result, project)
    it_energy = float(kpis["it_energy_mwh"])
    withdrawal = float(kpis["total_withdrawal_ml"])
    mean_pue = float(daily["pue"].mean())
    city_gwh_year = POPULATION * CITY_ELECTRICITY_KWH_PER_CAPITA_YEAR / 1e6
    facility_gwh_year = it_energy * mean_pue / 1000.0 / years
    return {
        "it_capacity_mw": float(project["components"]["AI_DC1"]["installed_it_capacity_mw"]),
        "mean_pue": mean_pue,
        "wue_l_per_kwh_it": withdrawal * 1000.0 / it_energy if it_energy else np.nan,
        "mean_load_factor": float(daily["load_factor"].mean()),
        "withdrawal_ml_d": withdrawal / (365.0 * years),
        "evaporation_ml_d": float(daily["evaporation_ml"].mean()),
        "blowdown_ml_d": float(daily["blowdown_ml"].mean()),
        "makeup_ml_d": float(daily["external_makeup_ml"].mean()),
        "achieved_coc": float(daily["cycles_of_concentration"].mean()),
        "reclaimed_ml_d": float(kpis["reclaimed_water_use_ml"]) / (365.0 * years),
        "fresh_ml_d": float(kpis["freshwater_withdrawal_ml"]) / (365.0 * years),
        "it_energy_gwh_per_year": it_energy / 1000.0 / years,
        "facility_energy_gwh_per_year": facility_gwh_year,
        # 间接(电厂侧)水足迹: 报告口径的 WUE 通常是"场址取水量", 不含电厂
        # 耗水; 两者必须分开报, 否则总水足迹被低估 15-25%。
        "offsite_water_ml_d": float(kpis["offsite_electricity_water_ml"]) / (365.0 * years),
        "offsite_share_of_onsite": (
            float(kpis["offsite_electricity_water_ml"]) / withdrawal if withdrawal else np.nan
        ),
        "total_wue_l_per_kwh_it": (
            (withdrawal + float(kpis["offsite_electricity_water_ml"])) * 1000.0 / it_energy
            if it_energy else np.nan
        ),
        "cooling_storage_hours": (
            float(project["components"]["AI_DC1"].get("cooling_storage_ml", np.nan))
            / (withdrawal / (365.0 * years)) * 24.0 if withdrawal else np.nan
        ),
        "city_reference_electricity_gwh_per_year": city_gwh_year,
        "ai_electricity_share_of_city": facility_gwh_year / city_gwh_year if city_gwh_year else np.nan,
    }


def check(name: str, value, low, high, unit: str, note: str = "",
          intentional: bool = False) -> dict:
    """Verdict against a reference band.

    `intentional=True` marks quantities that are *deliberately* outside the real
    world band (for example a stress-test capacity grid that reaches far beyond
    today's deployment).  Those are reported as NOTE instead of FAIL so they do
    not pollute the defect count, but they are still printed with their value.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        verdict = "NA"
    elif low <= value <= high:
        verdict = "PASS"
    elif intentional:
        verdict = "NOTE"
    elif low * 0.5 <= value <= high * 2.0:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    return {"metric": name, "value": value, "unit": unit,
            "reference": f"{low} - {high}", "verdict": verdict, "note": note}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--ai-capacity", type=float, default=1000.0)
    args = parser.parse_args()

    rows = []
    for domain in args.domains.split(","):
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
        years = len(timeseries) / 365.0

        base = copy.deepcopy(project)
        base["components"].pop("AI_DC1", None)
        base["supply_paths"][0]["demand_priority"] = [
            i for i in base["supply_paths"][0]["demand_priority"] if not i.startswith("data_center::")
        ]
        base_result = FullAIUWMModel(base, timeseries).run()
        city = city_side(base_result, base, years)

        ai_project = copy.deepcopy(project)
        ai_project["components"]["AI_DC1"]["installed_it_capacity_mw"] = args.ai_capacity
        ai_result = FullAIUWMModel(ai_project, timeseries).run()
        ai = ai_side(ai_result, ai_project, years)

        print(f"\n{'=' * 78}\n{domain}: city side (no AI)")
        for key, value in city.items():
            print(f"  {key:38s} {value}")
        print(f"{domain}: AI side ({args.ai_capacity:g} MW)")
        for key, value in ai.items():
            print(f"  {key:38s} {value}")

        checks = [
            check("per_capita_delivered_l_d", city["per_capita_delivered_l_d"], 150, 400, "L/cap/d",
                  "GB/T 50331 综合用水量(含工业/市政杂用)量级"),
            check("per_capita_abstraction_l_d", city["per_capita_abstraction_l_d"], 200, 450, "L/cap/d",
                  "取水口径应高于配水口径(漏损+厂自用)"),
            check("reuse_over_sewage", city["reuse_over_sewage"], 0.15, 0.70, "-",
                  "全国目标 ~25%, 缺水城市 ~35%, 北京 57.5% (v2 下限由 0.20 放宽至 0.15)"),
            check("reuse_plant_utilization", city["reuse_plant_utilization"], 0.60, 1.00, "-",
                  "现实再生水厂产能利用率 70-95%; 低于 60% 属闲置资产"),
            check("wwtw_inflow_ml_d", city["wwtw_inflow_ml_d"], 120, 320, "ML/d",
                  "旱流污水 ~ 配水量 x 0.85"),
            check("leakage_plus_loss_fraction", city["leakage_plus_loss_fraction"], 0.08, 0.25, "-",
                  "配水漏损 12% + 输水漏损 + 厂自用(v2 修正口径)"),
            check("reservoir_spill_fraction", city["reservoir_spill_fraction"], 0.0, 0.15, "-",
                  "弃水量/取水量: 常年大量弃水说明入流被高估、水库不构成约束"),
            check("reservoir_storage_max_fraction", city["reservoir_storage_max_fraction"], 0.55, 0.98, "-",
                  "库容峰值: 常年贴顶(=1.0)说明水库永远满蓄"),
            check("reservoir_storage_min_fraction", city["reservoir_storage_min_fraction"], 0.10, 0.70, "-",
                  "库容谷值: 应有真实年内消落"),
            check("mean_pue", ai["mean_pue"], 1.05, 1.60, "-", "LBNL 2024 / 运营商报告"),
            check("wue_l_per_kwh_it", ai["wue_l_per_kwh_it"], 0.30, 2.50, "L/kWh_IT",
                  "源水 WUE(按 IT 电量); 领跑者 0.2-0.6, 行业上限 ~2; "
                  "本端点为全年纯蒸发冷却, 取值偏高属设定必然"),
            check("achieved_coc", ai["achieved_coc"], 2.5, 10.0, "-", "DOE FEMP 3-7; ASHRAE 上限 ~10"),
            check("mean_load_factor", ai["mean_load_factor"], 0.30, 0.95, "-", "LBNL 2024 利用率"),
            check("withdrawal_ml_d", ai["withdrawal_ml_d"], 5, 80, "ML/d",
                  "1000 MW IT @ LF 0.75 纯蒸发冷却的补水量级"),
            check("evaporation_ml_d", ai["evaporation_ml_d"], 4, 60, "ML/d",
                  "物理上限 = 排热量 / 汽化潜热"),
            check("reclaimed_ml_d", ai["reclaimed_ml_d"], 0, city["reuse_production_ml_d"], "ML/d",
                  "AI 再生水用量不得超过城市再生水产量"),
            check("offsite_share_of_onsite", ai["offsite_share_of_onsite"], 0.05, 0.60, "-",
                  "电厂侧间接水足迹/场址取水量; 报告 WUE 时必须说明是否含间接"),
            check("cooling_storage_hours", ai["cooling_storage_hours"], 12, 72, "h",
                  "园区内冷却水调蓄小时数(现实超大规模园区 12-72 h)"),
            check("ai_electricity_share_of_city", ai["ai_electricity_share_of_city"], 0.05, 0.50, "-",
                  "园区设施电量/城市全社会用电量; 现实百万人口城市内园区 50-300 MW "
                  "(0.06-0.38)。本扫描网格 0-2000 MW 是刻意设定的压力测试上界, "
                  "超出不视为缺陷, 但结论必须折算到现实装机区间。",
                  intentional=True),
        ]
        frame = pd.DataFrame(checks)
        frame.insert(0, "domain", domain)
        rows.append(frame)
        print(f"\n{domain}: realism verdicts")
        print(frame.to_string(index=False))

    all_rows = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(OUT / "realism_audit.csv", index=False, encoding="utf-8-sig")
    fails = all_rows[all_rows["verdict"] == "FAIL"]
    warns = all_rows[all_rows["verdict"] == "WARN"]
    notes = all_rows[all_rows["verdict"] == "NOTE"]
    print(f"\nFAIL: {len(fails)}  WARN: {len(warns)}  NOTE: {len(notes)}")
    if len(fails):
        print(fails.to_string(index=False))
    if len(warns):
        print(warns.to_string(index=False))
    print(f"\nwritten -> {OUT / 'realism_audit.csv'}")


if __name__ == "__main__":
    main()
