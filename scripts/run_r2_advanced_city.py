"""R2 边界条件实验：把模型城市改成"政策已达标"的先进城市，看结论还成不成立。

**动机（本轮自己标出的局限 #16）**
12.6 节的城市侧对标显示：模型城市的漏损率 12–13.3 % 对照 2025 年 9 % 政策目标
**超标**，再生水利用率 15–16 % 对照十四五目标 25 % **未达标**。而 10.4 节
"降漏损是 D-HA 唯一有效干预"与 12.3 节"政策约束先于物理约束触发"两条结论
**都建立在这个落后设定之上**。若不检验边界，读者可以合理地怀疑：
这些结论只是"落后城市"的产物，对已完成治理的城市无效。

本脚本直接把城市改到达标，重跑同一套容量扫描与干预边际，给出结论的适用边界。

**四个变体（两域各四）**
  baseline    现状城市（漏损 0.12、再生水政策利用率 ~15–16 %）
  leakage_09  仅漏损治理达标：DM1 leakage_fraction 0.12 → 0.09（2025 政策目标）
  reuse_25    仅再生水达标：政策口径利用率提到 25 %（改分流比例 + 配套处理能力）
  both        两项同时达标

**观测**
  1. 日/年尺度 CAWCC 与限制约束（结论 1：政策约束是否还主导）
  2. leakage_minus_30pct 与 grid_plus_50pct 的边际增益（结论 2：对症干预是否还有效）
  3. 城市侧指标（漏损率、人均用水、再生水利用率）复核，确认变体真的达标了

用法:
    PYTHONPATH="src;scripts" python scripts/run_r2_advanced_city.py
    PYTHONPATH="src;scripts" python scripts/run_r2_advanced_city.py --domains ha
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_r2 import (  # noqa: E402
    CAPACITIES_SCAN,
    R2,
    annual_constraints,
    daily_constraints,
    mean_utilizations,
    peak_day_factor,
    scan_with_mean,
)

from aiuwm.ai_capacity import find_ai_carrying_capacity  # noqa: E402
from aiuwm.ai_metrics import summarize_ai_water_kpis  # noqa: E402
from aiuwm.full_engine import FullAIUWMModel  # noqa: E402

OUT = ROOT / "validation_artifacts" / "r2"

DOMAIN_LABEL = {"ha": "D-HA 缺水—高冷负荷端", "co": "D-CO 气候凉爽端"}
VARIANT_LABEL = {
    "baseline": "现状城市（漏损 12 %、再生水利用率 ~15–16 %）",
    "leakage_09": "仅漏损达标（0.12 → 0.09，2025 政策目标）",
    "reuse_25": "仅再生水达标（政策口径利用率 → 25 %）",
    "both": "两项同时达标",
}

# 政策口径再生水利用率目标 = 再生水利用量 / 污水处理量
REUSE_POLICY_TARGET = 0.25
# 现状 → 目标所需的分流比例放大系数（先按现状实测比例反算，见 _scale_reuse）
LEAKAGE_TARGET = 0.09


def _base_materials(domain: str):
    project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
    constraints = json.loads(
        (R2 / "specs" / f"constraints_{domain}.json").read_text(encoding="utf-8")
    )
    return project, timeseries, constraints


def _measure_policy_reuse(project: dict, timeseries: pd.DataFrame) -> float:
    """按政策口径（再生水利用量 / 污水处理量）测现状再生水利用率。"""
    result = FullAIUWMModel(copy.deepcopy(project), timeseries).run()
    comp = result.component_daily
    if comp.empty:
        return float("nan")
    agg = comp.groupby("component_id")["treated_ml"].sum()
    wwtw = float(agg.get("WWTW1", 0.0))
    reuse = float(agg.get("CENTRAL_REUSE", 0.0))
    return reuse / wwtw if wwtw else float("nan")


def _make_variant(project: dict, timeseries: pd.DataFrame, variant: str,
                  current_reuse_ratio: float) -> dict:
    trial = copy.deepcopy(project)
    if variant in ("leakage_09", "both"):
        trial["components"]["DM1"]["leakage_fraction"] = LEAKAGE_TARGET
    if variant in ("reuse_25", "both"):
        # 分流比例按"现状政策口径利用率 → 目标值"等比放大
        scale = REUSE_POLICY_TARGET / current_reuse_ratio if current_reuse_ratio > 0 else 1.0
        wwtw = trial["components"]["WWTW1"]
        wwtw["central_reuse_fraction"] = min(
            float(wwtw["central_reuse_fraction"]) * scale, 0.95
        )
        # 处理能力必须同步放大，否则新增的分流水在厂门口被卡住（10.3 节的教训）
        trial["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"] = (
            float(trial["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"]) * scale
        )
    return trial


def _intervention_gain(project: dict, timeseries: pd.DataFrame, constraints: dict,
                       specs: dict, kind: str) -> dict:
    """在给定城市上测单个干预的 CAWCC 增益（与 10.2 节同一口径：读干预规格逐条 set）。"""
    day_constraints = {"constraints": daily_constraints(constraints)}

    def cawcc_of(cfg: dict) -> dict:
        scan = scan_with_mean(cfg, timeseries, CAPACITIES_SCAN)
        positive = scan[scan["ai_capacity_mw"] >= 50].reset_index(drop=True)
        return find_ai_carrying_capacity(positive, day_constraints)

    spec = next((i for i in specs["interventions"] if i["name"] == kind), None)
    if spec is None:
        return {"CAWCC_日_MW": None, "限制约束": None, "增益_MW": None, "备注": "干预规格缺失"}

    if kind == "baseline":
        base = cawcc_of(project)
        return {"CAWCC_日_MW": base["maximum_safe_ai_capacity_mw"],
                "限制约束": base["limiting_constraint"], "增益_MW": 0.0, "备注": ""}

    trial = copy.deepcopy(project)
    for entry in spec["set"]:
        target = trial
        parts = entry["path"].split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = entry["value"]

    base = cawcc_of(project)
    after = cawcc_of(trial)
    return {
        "CAWCC_日_MW": after["maximum_safe_ai_capacity_mw"],
        "限制约束": after["limiting_constraint"],
        "增益_MW": float(after["maximum_safe_ai_capacity_mw"])
        - float(base["maximum_safe_ai_capacity_mw"]),
        "备注": "",
    }


def _city_indicators(project: dict, timeseries: pd.DataFrame) -> dict:
    """复核变体城市的关键城市侧指标（与 12.6 节同口径）。"""
    result = FullAIUWMModel(copy.deepcopy(project), timeseries).run()
    comp = result.component_daily
    system = result.system_daily
    area = result.area_daily
    agg = comp.groupby("component_id")[["inflow_ml", "leakage_ml", "treated_ml"]].sum()

    def loss(cid: str) -> float | None:
        if cid not in agg.index or agg.loc[cid, "inflow_ml"] <= 0:
            return None
        return float(agg.loc[cid, "leakage_ml"] / agg.loc[cid, "inflow_ml"] * 100.0)

    survive = 1.0
    for value in (loss("SC1"), loss("TM1"), loss("DM1")):
        if value is not None:
            survive *= (1.0 - value / 100.0)
    wwtw = float(agg.loc["WWTW1", "treated_ml"]) if "WWTW1" in agg.index else 0.0
    reuse = float(agg.loc["CENTRAL_REUSE", "treated_ml"]) if "CENTRAL_REUSE" in agg.index else 0.0
    days = float(len(system))
    pop = float(area["population"].iloc[-1])
    return {
        "配水干管漏损率_pct": loss("DM1"),
        "输配全链漏损率_pct": (1.0 - survive) * 100.0,
        "再生水政策口径利用率_pct": reuse / wwtw * 100.0 if wwtw else None,
        "人均生活用水_平均日_L_per_cap_d": float(
            area.groupby("date")["demand_domestic_ml"].sum().mean()
        ) * 1e6 / pop,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--variants", default="baseline,leakage_09,reuse_25,both")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    cawcc_rows: list[dict] = []
    intervention_rows: list[dict] = []
    indicator_rows: list[dict] = []

    for domain in domains:
        project, timeseries, constraints = _base_materials(domain)
        specs = json.loads(
            (R2 / "specs" / f"interventions_{domain}.json").read_text(encoding="utf-8")
        )
        current_reuse = _measure_policy_reuse(project, timeseries)
        print(f"[{domain}] 现状政策口径再生水利用率 = {current_reuse * 100:.2f} %", flush=True)

        for variant in variants:
            cfg = _make_variant(project, timeseries, variant, current_reuse)
            scan = scan_with_mean(cfg, timeseries, CAPACITIES_SCAN)
            positive = scan[scan["ai_capacity_mw"] >= 50].reset_index(drop=True)
            day = {"constraints": daily_constraints(constraints)}
            daily = find_ai_carrying_capacity(positive, day)
            annual = find_ai_carrying_capacity(
                positive, {"constraints": annual_constraints(constraints)}
            )

            cawcc_rows.append({
                "域": domain,
                "域标签": DOMAIN_LABEL[domain],
                "变体": variant,
                "变体说明": VARIANT_LABEL[variant],
                "CAWCC_年_MW": annual["maximum_safe_ai_capacity_mw"],
                "年尺度限制约束": annual["limiting_constraint"],
                "CAWCC_日_MW": daily["maximum_safe_ai_capacity_mw"],
                "日尺度限制约束": daily["limiting_constraint"],
            })

            ind = _city_indicators(cfg, timeseries)
            indicator_rows.append({"域": domain, "域标签": DOMAIN_LABEL[domain],
                                   "变体": variant, **ind})

            for kind in ("baseline", "leakage_minus_30pct", "grid_plus_50pct"):
                got = _intervention_gain(cfg, timeseries, constraints, specs, kind)
                intervention_rows.append({
                    "域": domain,
                    "域标签": DOMAIN_LABEL[domain],
                    "变体": variant,
                    "干预": kind,
                    **got,
                })
            print(f"  [{domain}/{variant}] 日 CAWCC {daily['maximum_safe_ai_capacity_mw']:.0f} MW"
                  f" ({daily['limiting_constraint']})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cawcc = pd.DataFrame(cawcc_rows)
    inter = pd.DataFrame(intervention_rows)
    ind = pd.DataFrame(indicator_rows)
    cawcc.to_csv(OUT / "advanced_city_cawcc.csv", index=False, encoding="utf-8-sig")
    inter.to_csv(OUT / "advanced_city_interventions.csv", index=False, encoding="utf-8-sig")
    ind.to_csv(OUT / "advanced_city_indicators.csv", index=False, encoding="utf-8-sig")

    print("\n=== 已达标城市：CAWCC 与限制约束 ===")
    print(cawcc[["域", "变体", "CAWCC_年_MW", "年尺度限制约束",
                 "CAWCC_日_MW", "日尺度限制约束"]].to_string(index=False))
    print("\n=== 干预增益（同一个干预在不同城市上的效果）===")
    print(inter[["域", "变体", "干预", "CAWCC_日_MW", "增益_MW", "限制约束"
                 ]].to_string(index=False))
    print("\n=== 城市侧指标复核 ===")
    print(ind.to_string(index=False))
    print(f"\n-> {OUT / 'advanced_city_cawcc.csv'}")
    print(f"-> {OUT / 'advanced_city_interventions.csv'}")
    print(f"-> {OUT / 'advanced_city_indicators.csv'}")


if __name__ == "__main__":
    main()
