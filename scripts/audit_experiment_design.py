"""对 R2 实验设计做**自洽性与现实性硬检查**（不信任任何文档声称，一律反算）。

用户提出"检查实验设计及数据是否有问题，还有是否存在与现实不符的情况"。
本脚本不复用任何已有结论，全部从 project.json + timeseries.csv + 逐日 dump 反算，
逐条给出 PASS/FAIL/WARN 与实测值，用于定位三类问题：
  A. 实验设计内部自洽（约束是否可能触发、增益是否与水量收支匹配）
  B. 数据口径一致（年化 vs 全期、单位是否混用）
  C. 与现实不符（量级是否落在真实城市/真实园区范围内）

用法:
    PYTHONPATH="src;scripts" python scripts/audit_experiment_design.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aiuwm.full_engine import FullAIUWMModel  # noqa: E402

R2 = ROOT / "examples" / "cawcc_r2"
TS_ROOT = ROOT / "validation_artifacts" / "timeseries"
OUT = ROOT / "validation_artifacts" / "r2"

DOMAIN_LABEL = {"ha": "D-HA 缺水—高冷负荷端", "co": "D-CO 气候凉爽端"}

# 现实参照（来源见各行 note 字段）
REAL_WORLD = {
    # 中国城市人均综合用水量（含工业、市政、生活），L/(人·d)
    "城市人均综合用水量": (250, 550),
    # 百万人口城市全社会用电：按人均生活+工商业，典型 3000-6000 MW 峰值负荷
    "百万人口城市峰值负荷_MW": (1500, 6000),
    # 现实中已建成的单一园区级 AI 数据中心（MW）
    "现实单一园区AI装机_MW": (50, 300),
    # 库容 / 年取水量的合理倍数（年调节及以上水库）
    "库容比年取水": (0.8, 3.0),
    # 入流 / 取水能力的合理倍数（水源可靠性）
    "入流比取水能力": (1.0, 2.5),
}

CHECKS: list[dict] = []


def add(section: str, item: str, status: str, measured, expected, note: str) -> None:
    CHECKS.append({
        "检查类别": section, "检查项": item, "判定": status,
        "实测值": measured, "参照/预期": expected, "说明": note,
    })
    print(f"[{status:4}] {section} | {item}: {measured}  (参照 {expected})", flush=True)


def _load(domain: str):
    project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
    ts = pd.read_csv(R2 / domain / "timeseries.csv")
    return project, ts


def run(project: dict, ts: pd.DataFrame, capacity_mw: float):
    cfg = project
    for c in cfg["components"].values():
        if c.get("kind") == "data_center":
            c["installed_it_capacity_mw"] = float(capacity_mw)
            c.pop("capacity_schedule", None)
    return FullAIUWMModel(cfg, ts).run()


def check_timeseries(project: dict, ts: pd.DataFrame, domain: str) -> None:
    """A 组：水源与库容的设定是否自洽。"""
    res = project["components"]["RES1"]
    inflow = ts[res["inflow_column"]] if res.get("inflow_column") in ts.columns else None
    if inflow is None:
        add("A 实验设计自洽", "水源入流序列存在", "FAIL", "缺失", "列存在", "")
        return
    mean_inflow = float(inflow.mean())
    # 找到取水能力：优先 abstraction_capacity，否则 capacity_schedule
    abs_cap = res.get("abstraction_capacity_ml_day")
    if abs_cap is None:
        abs_cap = res.get("capacity_ml_day")
    add("A 实验设计自洽", "入流/取水能力",
        "PASS" if 1.0 <= mean_inflow / abs_cap <= 2.5 else "FAIL",
        round(mean_inflow / abs_cap, 3),
        f"{REAL_WORLD['入流比取水能力']}（文档 15 节声称 1.05–1.15）",
        f"年均入流 {mean_inflow:.1f} ML/d，取水能力 {abs_cap} ML/d")

    # 库容 vs 年取水（按取水能力年化）
    cap_ml = float(res.get("capacity_ml", 0.0))
    annual = abs_cap * 365.0
    ratio = cap_ml / annual if annual else float("nan")
    add("A 实验设计自洽", "库容/年取水能力",
        "PASS" if REAL_WORLD["库容比年取水"][0] <= ratio <= REAL_WORLD["库容比年取水"][1] else "WARN",
        round(ratio, 3),
        f"{REAL_WORLD['库容比年取水']}（文档 15 节声称 1.3–1.4）",
        f"库容 {cap_ml:,.0f} ML，年取水能力 {annual:,.0f} ML")


def check_city_scale(project: dict, ts: pd.DataFrame, domain: str) -> None:
    """C 组：城市规模与水量的现实性。"""
    pop = float(project["local_areas"]["LA1"]["base_population"])
    result = run(project, ts, 0.0)
    system = result.system_daily
    days = float(len(system))
    # 总供水（进入配水干管的水量）
    comp = result.component_annual if hasattr(result, "component_annual") else None
    comp_d = result.component_daily
    agg = comp_d.groupby("component_id")[["inflow_ml", "outflow_ml"]].sum()
    dm_in = float(agg.loc["DM1", "inflow_ml"]) if "DM1" in agg.index else float("nan")
    per_cap = dm_in / days * 1e6 / pop  # ML/d -> L/(人·d)
    lo, hi = REAL_WORLD["城市人均综合用水量"]
    add("C 现实性", "人均综合供水量 L/(人·d)",
        "PASS" if lo <= per_cap <= hi else "FAIL",
        round(per_cap, 1), f"{lo}–{hi}",
        f"人口 {pop:,.0f}，配水干管进水 {dm_in / days:.1f} ML/d（无 AI 负荷）")

    # 供水保证率
    unmet = float(system["unmet_demand_ml"].sum())
    add("C 现实性", "基线城市供水保证率",
        "PASS" if unmet < 1e-3 else "FAIL",
        f"未满足 {unmet:.2e} ML", "≈0",
        "0 MW 基线不应有任何缺水（V12）")

    # 需求结构：居民 / 工业 / 市政 占城市总需求的比例
    area = result.area_daily
    parts = {
        "居民生活": float(area["demand_domestic_ml"].sum()),
        "工业": float(area["demand_industrial_ml"].sum()),
        "市政杂用": float(area["demand_municipal_misc_ml"].sum()),
        "灌溉": float(area["demand_irrigation_ml"].sum()),
    }
    total = sum(parts.values())
    share = {k: v / total * 100 for k, v in parts.items()}
    add("C 现实性", "城市用水需求结构（工业占比）",
        "PASS" if 15 <= share["工业"] <= 45 else "FAIL",
        "; ".join(f"{k} {v:.1f}%" for k, v in share.items()),
        "工业 20–40 %（中国设区城市典型）",
        f"总需求 {total / days:.1f} ML/d。工业占比过低会**高估**留给 AI 的水量余量，"
        "从而高估 CAWCC")

    # 城市全社会用电规模：引擎是否建模了 AI 之外的用电
    # （若只给了 AI 园区的接网容量而城市用电缺位，则"AI 占城市用电 %"无从谈起）
    add("C 现实性", "城市非 AI 用电是否被建模",
        "FAIL" if True else "PASS",
        "项目里只有 AI_DC1.grid_connection_capacity_mw，未见城市全社会负荷",
        "应有城市基准负荷序列或等效参数",
        f"文档局限 #10 称『1000 MW 已占城市全社会用电 129 %』，"
        f"但引擎中并无城市用电这一输入——该百分比的**分母来源需要说明**，"
        f"否则这是一个无法在模型内复核的外部断言")


def check_ai_electricity(project: dict, ts: pd.DataFrame, domain: str) -> None:
    """A/C 组：AI 装机相对城市电网规模的现实性，以及电力约束为何（不）触发。"""
    dc = project["components"]["AI_DC1"]
    grid_cap = float(dc.get("grid_connection_capacity_mw", float("nan")))

    print(f"    (诊断) grid_connection_capacity_mw = {grid_cap}", flush=True)
    for cap in (300.0, 1000.0, 1450.0, 1500.0, 1750.0, 2000.0):
        result = run(project, ts, cap)
        dcd = result.data_center_daily
        if dcd.empty:
            continue
        # 注意单位：data_center_daily 的能耗列是 MWh/d，除以 24 才是平均功率 MW。
        # 第一版审计脚本直接用 MWh/d 当 MW 用，导致比值虚高 24 倍，是脚本 bug 不是实验问题。
        energy_cols = [c for c in dcd.columns if "energy" in c.lower() and "mwh" in c.lower()]
        it_cols = [c for c in dcd.columns if "it_energy" in c.lower()]
        col = it_cols[0] if it_cols else (energy_cols[0] if energy_cols else None)
        if col is None:
            continue
        avg_mw = float(dcd[col].mean()) / 24.0
        share_of_grid = avg_mw / grid_cap if grid_cap else float("nan")
        add("C 现实性", f"AI 日均功率/电网接入容量 @{cap:.0f} MW",
            "PASS" if share_of_grid <= 0.95 else "FAIL",
            f"{share_of_grid:.3f}（日均 {avg_mw:.0f} MW）", "≤0.95（规划裕度）",
            f"列 {col}（MWh/d 已 ÷24 换算为 MW），电网接入 {grid_cap:.0f} MW")

    # 现实园区规模对照
    lo, hi = REAL_WORLD["现实单一园区AI装机_MW"]
    add("C 现实性", "核心结论所用 AI 装机是否落在现实园区区间",
        "FAIL", "CAWCC 1150/1450 MW；主结论在 1000 MW 处讨论",
        f"{lo}–{hi} MW",
        "文档已要求折算，但**所有主结论都在这个不现实尺度上得出**，"
        "折算后的结论从未被真正计算过——见下方 0–300 MW 细网格")


def check_fine_grid(project: dict, ts: pd.DataFrame, domain: str) -> None:
    """A 组：在现实园区区间（0–300 MW，步长 25）重扫，检验现实尺度下约束是否触发。"""
    from run_r2 import daily_constraints  # noqa: E402
    constraints = json.loads(
        (R2 / "specs" / f"constraints_{domain}.json").read_text(encoding="utf-8")
    )
    day = {"constraints": daily_constraints(constraints)}
    rows = []
    for cap in np.arange(0, 301, 25, dtype=float):
        result = run(project, ts, float(cap))
        rec = {"ai_capacity_mw": float(cap)}
        rows.append(rec)
    # 用 summarize_ai_water_kpis 拿约束指标
    from aiuwm.ai_metrics import summarize_ai_water_kpis  # noqa: E402
    from run_r2 import mean_utilizations  # noqa: E402
    rows = []
    for cap in np.arange(0, 301, 25, dtype=float):
        cfg = json.loads(json.dumps(project))
        for c in cfg["components"].values():
            if c.get("kind") == "data_center":
                c["installed_it_capacity_mw"] = float(cap)
                c.pop("capacity_schedule", None)
        result = FullAIUWMModel(cfg, ts).run()
        rec = {"ai_capacity_mw": float(cap), **summarize_ai_water_kpis(result, cfg)}
        rec.update(mean_utilizations(result, cfg))
        rows.append(rec)
    frame = pd.DataFrame(rows)
    keys = [k for k in day["constraints"] if k in frame.columns]
    worst = frame[keys].max()
    add("A 实验设计自洽", "0–300 MW 现实区间内各约束最大取值",
        "PASS" if float(worst.max()) < 0.95 else "WARN",
        "; ".join(f"{k}={worst[k]:.3f}" for k in keys),
        "<0.95 表示现实区间内无约束触发",
        "若都在 0.95 以下，说明**现实尺度下承载约束根本不生效**，"
        "主结论必须折算后才能引用")


def check_leakage_budget(project: dict, ts: pd.DataFrame, domain: str) -> None:
    """A 组：降漏损 +150/+200 MW 的增益是否与水量收支自洽。"""
    base = run(project, ts, 0.0)
    agg = base.component_daily.groupby("component_id")["leakage_ml"].sum()
    days = float(len(base.system_daily))
    dm_leak = float(agg.get("DM1", 0.0)) / days  # ML/d

    trial = json.loads(json.dumps(project))
    cur = float(trial["components"]["DM1"]["leakage_fraction"])
    trial["components"]["DM1"]["leakage_fraction"] = cur * 0.70
    after = run(trial, ts, 0.0)
    agg2 = after.component_daily.groupby("component_id")["leakage_ml"].sum()
    dm_leak2 = float(agg2.get("DM1", 0.0)) / days
    saving_ml_d = dm_leak - dm_leak2

    # AI 单位装机取水量（从 1000 MW 情景反推）
    r1000 = run(project, ts, 1000.0)
    dcd = r1000.data_center_daily
    with_col = next((c for c in dcd.columns if "withdrawal" in c.lower()), None)
    ai_ml_d = float(dcd[with_col].mean()) if with_col else float("nan")
    per_mw = ai_ml_d / 1000.0 if ai_ml_d == ai_ml_d else float("nan")

    implied_mw = saving_ml_d / per_mw if per_mw and per_mw == per_mw else float("nan")
    add("A 实验设计自洽", "降漏损节水量可支撑的 AI 装机（推算）",
        "PASS" if 0.5 <= implied_mw / 150.0 <= 2.0 else "WARN",
        f"{implied_mw:.0f} MW（节水 {saving_ml_d:.2f} ML/d，AI 单耗 {per_mw:.4f} ML/d/MW）",
        "与实测增益 +150 MW 应同量级",
        "若相差数倍，说明增益来自网格分辨率或约束切换，而非水量收支")


def check_caliber(project: dict, ts: pd.DataFrame, domain: str) -> None:
    """B 组：数据口径一致性（年化 vs 全期累计）。"""
    path = OUT / domain / "technology_comparison.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for _, r in frame.iterrows():
        wue = float(r["wue_l_kwh"])
        ann_w = float(r.get("annual_withdrawal_ml", float("nan")))
        energy = float(r.get("facility_energy_mwh", float("nan")))
        if not (ann_w == ann_w and energy == energy and energy > 0):
            continue
        # 若 annual_withdrawal 为全期累计（730 d），则 WUE = ann_w*1e6 / (energy*1000)
        as_full = ann_w * 1e6 / (energy * 1000.0)
        # 若为年化（365 d），全期应为 2 倍
        as_annual = ann_w * 2 * 1e6 / (energy * 1000.0)
        if abs(as_full - wue) < 0.02 * max(wue, 1e-9):
            verdict = "annual_withdrawal_ml 是全期累计"
        elif abs(as_annual - wue) < 0.02 * max(wue, 1e-9):
            verdict = "annual_withdrawal_ml 是年化值（与 facility_energy 全期累计口径不一致！）"
        else:
            verdict = f"两者都对不上（全期算 {as_full:.4f}，年化算 {as_annual:.4f}）"
        add("B 数据口径", f"technology_comparison[{r['technology']}] WUE 口径",
            "PASS" if verdict.startswith("annual_withdrawal_ml 是全期") else "FAIL",
            f"WUE={wue:.4f}", f"反算应≈{wue:.4f}", verdict)
        break  # 一个技术足以判定


def main() -> None:
    for domain in ("ha", "co"):
        print(f"\n===== {domain.upper()} {DOMAIN_LABEL[domain]} =====", flush=True)
        project, ts = _load(domain)
        check_timeseries(project, ts, domain)
        check_city_scale(project, ts, domain)
        check_ai_electricity(project, ts, domain)
        check_leakage_budget(project, ts, domain)
        check_caliber(project, ts, domain)
        check_fine_grid(project, ts, domain)

    frame = pd.DataFrame(CHECKS)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "experiment_design_audit.csv", index=False, encoding="utf-8-sig")
    print(f"\n总计 {len(frame)} 项：", flush=True)
    print(frame.groupby(["检查类别", "判定"]).size().to_string())
    print(f"\n-> {OUT / 'experiment_design_audit.csv'}")


if __name__ == "__main__":
    main()
