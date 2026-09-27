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
    # 城市配水口径（不含农业和生态补水）的公开城市量级，L/(人·d)。
    # 这不是全国统一阈值；综合用水量和居民生活用水不能混用。
    "城市人均综合用水量": (150, 550),
    # 百万人口城市全社会用电：按人均生活+工商业，典型 3000-6000 MW 峰值负荷
    "百万人口城市峰值负荷_MW": (1500, 6000),
    # 该量不设固定上限。公开项目已出现 1 GW 级 AI 园区，50–300 MW
    # 只能作为小型园区情景，不能作为“现实单园区”的硬边界。
    "现实单一园区AI装机_MW": (50, 1800),
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
        "WARN" if mean_inflow < abs_cap else "PASS",
        round(mean_inflow / abs_cap, 3),
        "若长期实际取水超过入流，需检验库容趋势（能力是上限，不是日均取水）",
        f"年均入流 {mean_inflow:.1f} ML/d，取水能力 {abs_cap} ML/d；此处是稳态诊断，不是现实性判否")

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

    # 城市全社会用电规模：把非 AI 基线作为显式驱动列，避免把 AI
    # 园区接入容量误当作城市总负荷。该列只用于城市电力占比和节点余量
    # 诊断，不改变 AI 园区自身的接网约束。
    column = str(project.get("city_context", {}).get(
        "non_ai_load_column", "city_non_ai_electricity_mw"
    ))
    if column in ts and (ts[column].astype(float) > 0).all():
        base = ts[column].astype(float)
        add("C 现实性", "城市非 AI 用电是否被建模", "PASS",
            f"均值 {base.mean():.1f} MW；峰值 {base.max():.1f} MW",
            "存在正值城市基线负荷序列",
            f"来源列 {column}；用于 AI 占城市负荷和节点余量诊断")
    else:
        add("C 现实性", "城市非 AI 用电是否被建模", "FAIL",
            f"缺失或非正值列 {column}",
            "存在正值城市基线负荷序列",
            "城市电力占比无法由接网容量单独推导")


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
        # 接网容量约束对应设施从电网取得的平均功率，应使用 facility
        # energy；IT energy 会低估 PUE>1 时的接网占用。
        facility_cols = [c for c in dcd.columns if "facility_energy" in c.lower()]
        col = facility_cols[0] if facility_cols else (energy_cols[0] if energy_cols else None)
        if col is None:
            continue
        mean_mw = float(dcd[col].mean()) / 24.0
        peak_mw = float(dcd[col].max()) / 24.0
        share_of_grid = peak_mw / grid_cap if grid_cap else float("nan")
        add("C 现实性", f"AI 峰值日均功率/电网接入容量 @{cap:.0f} MW",
            "PASS" if share_of_grid <= 0.95 else "FAIL",
            f"{share_of_grid:.3f}（均值 {mean_mw:.0f} MW，峰值 {peak_mw:.0f} MW）", "≤0.95（规划裕度）",
            f"列 {col}（MWh/d ÷24；约束采用逐日峰值），电网接入 {grid_cap:.0f} MW")

    # 现实园区规模只作情景标签，不把某个固定区间当作现实上限。
    add("C 现实性", "压力测试容量是否有公开部署尺度参照",
        "PASS", "50–2000 MW；公开项目已出现约 1 GW 级 AI 园区",
        "参照项目和电网条件应逐项目注明",
        "50–300 MW 可代表中型园区；1 GW 级项目已进入公开规划，容量本身不能判定为失真")


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


def check_reservoir_sustainability(domain: str) -> None:
    """Separate short-term facility throughput from end-storage sustainability."""
    path = OUT / "reservoir_sustainability_corrected.json"
    if not path.exists():
        add("A 实验设计自洽", "末库容可持续性扫描", "WARN", "尚未生成",
            "应报告期末库容不低于初始库容的容量边界",
            "运行 scripts/audit_reservoir_sustainability.py 后纳入当前结果")
        return
    report = json.loads(path.read_text(encoding="utf-8")).get(domain, {})
    add("A 实验设计自洽", "末库容不低于初始值的最大容量", "WARN",
        f"{report.get('max_terminal_non_depleting_capacity_mw')} MW（首次下降 {report.get('first_terminal_decline_capacity_mw')} MW）",
        "与设施吞吐 CAWCC 分开报告",
        "两年合成驱动诊断；不等同多年可靠供水能力")


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
    positive_rows = frame[frame["wue_l_kwh"].astype(float) > 0]
    for _, r in positive_rows.head(1).iterrows():
        wue = float(r["wue_l_kwh"])
        ann_w = float(r.get("annual_withdrawal_ml", float("nan")))
        # DOE/ISO 的 site WUE 分母是 IT 能耗；此表原先仅给设施能耗。
        # 用同一技术情景 750 MW 的 IT 负荷逐日积分可直接反算。
        tech_project = json.loads((R2 / domain / f"tech_{r['technology']}.json").read_text(encoding="utf-8"))
        tech_project["components"]["AI_DC1"]["installed_it_capacity_mw"] = 750.0
        tech_result = FullAIUWMModel(tech_project, ts).run()
        energy = float(tech_result.data_center_daily["it_energy_mwh"].sum())
        if not (ann_w == ann_w and energy == energy and energy > 0):
            continue
        # ML 和 MWh 的单位比为 1000 L/kWh；两年期为年值的 2 倍。
        as_full = ann_w * 1000.0 / energy
        as_annual = ann_w * 2 * 1000.0 / energy
        if abs(as_full - wue) < 0.02 * max(wue, 1e-9):
            verdict = "annual_withdrawal_ml 是全期累计"
        elif abs(as_annual - wue) < 0.02 * max(wue, 1e-9):
            verdict = "annual_withdrawal_ml 是年化值；WUE 使用同一期的全期取水与 IT 能耗"
        else:
            verdict = f"两者都对不上（全期算 {as_full:.4f}，年化算 {as_annual:.4f}）"
        add("B 数据口径", f"technology_comparison[{r['technology']}] WUE 口径",
            "PASS" if verdict.startswith("annual_withdrawal_ml 是年化") else "FAIL",
            f"WUE={wue:.4f}", f"反算应≈{wue:.4f}", verdict)


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
        check_reservoir_sustainability(domain)

    frame = pd.DataFrame(CHECKS)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "experiment_design_audit.csv", index=False, encoding="utf-8-sig")
    print(f"\n总计 {len(frame)} 项：", flush=True)
    print(frame.groupby(["检查类别", "判定"]).size().to_string())
    print(f"\n-> {OUT / 'experiment_design_audit.csv'}")


if __name__ == "__main__":
    main()
