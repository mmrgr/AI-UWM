"""R2 干预方案的成本效益与多目标 Pareto 前沿（任务 D）。

前面的实验只回答了"哪个干预能多扛多少 MW"，没有回答"值不值"。
本脚本把每个工程干预换算成投资额，建立：
    投资(万元)  vs  CAWCC 增益(MW)  vs  居民未满足量增量(ML)
的三维 Pareto 前沿，识别"性价比拐点"与"绝对无效项"。

成本参数全部带来源，来源性质分三档（诚实标注，不冒充学术文献）：
  OFFICIAL  政府/电网公司正式文件
  CASE      公开招投标或核准批复的具体项目
  GREY      咨询机构或厂商网页（量级参考，论文引用前必须替换为正式来源）

水量单位换算：模型输出 ML/d，造价指标为 元/(m³/d)，故 ML/d -> m³/d 乘 1000。

用法:
    PYTHONPATH="src;scripts" python scripts/run_r2_cost_pareto.py
"""
from __future__ import annotations

import copy
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"

DOMAIN_LABEL = {"ha": "D-HA 缺水—高冷负荷端", "co": "D-CO 气候凉爽端"}

# ---------------------------------------------------------------------------
# 成本参数表：单位造价区间 + 来源 + 来源性质
# ---------------------------------------------------------------------------
COST_PARAMS = [
    {
        "参数": "给水厂扩容增量单位投资",
        "取值低": 2170.0, "取值高": 2715.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "wtw_plus_*",
        "来源": "《污水厂、给水厂及垃圾处理设施造价参考指标》(询价库造价咨询，2025-07，重庆)：给水厂每增加 10000 m³/d 投资 2170–2715 万元",
        "来源性质": "GREY",
    },
    {
        "参数": "污水处理厂扩容增量单位投资",
        "取值低": 1630.0, "取值高": 2170.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "wwtw_plus_*",
        "来源": "同上：普通污水处理厂每增加 10000 m³/d 投资 1630–2170 万元",
        "来源性质": "GREY",
    },
    {
        "参数": "中水回用单位投资",
        "取值低": 5000.0, "取值高": 7000.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "reuse_plus_*",
        "来源": "同上：中水回用 5000–7000 元/(m³/d)",
        "来源性质": "GREY",
    },
    {
        "参数": "饮用水处理设施单位投资",
        "取值低": 2000.0, "取值高": 3500.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "（交叉校验用）",
        "来源": "《水厂类项目单位造价参考体系》(询价无忧网，2023 价格水平)：饮用水处理 2000–3500 元/m³",
        "来源性质": "GREY",
    },
    {
        "参数": "常规污水处理单位投资",
        "取值低": 2500.0, "取值高": 4500.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "（交叉校验用）",
        "来源": "同上：日处理 <10 万 m³ 常规污水处理 2500–4500 元/m³",
        "来源性质": "GREY",
    },
    {
        "参数": "110kV 变电站单位造价（静态控制线）",
        "取值低": 625.0, "取值高": 625.0, "单位": "元/kVA",
        "适用干预": "grid_plus_*",
        "来源": "《35–500 千伏输变电工程造价控制线(2022年)》南方电网办基建〔2022〕28号：110kV 新建 1×63MVA GIS 静态 625 元/kVA",
        "来源性质": "OFFICIAL",
    },
    {
        "参数": "数据中心专用 110kV 变电站单位投资（案例区间）",
        "取值低": 369.0, "取值高": 769.0, "单位": "元/kVA",
        "适用干预": "grid_plus_*",
        "来源": "中国移动呼和浩特数据中心第二 110kV 站 4×63MVA 总投资 9287 万元(呼和浩特市发改委核准批复 2025-07-30，约 369 元/kVA)；"
                "兴业银行贵安数据中心 2×63MVA EPC 中标 9692.78 万元(贵州公共资源交易 2025-06-30，约 769 元/kVA)",
        "来源性质": "CASE",
    },
    {
        "参数": "漏损治理单位节水投资",
        "取值低": 5.38, "取值高": 10.0, "单位": "元/(m³·a⁻¹)",
        "适用干预": "leakage_minus_*",
        "来源": "案例一：某老城区 DMA+压力调控+声学定位，投资 860 万元、年节水 86 万 m³ → 10.0 元/(m³·a)；"
                "案例二：南方某地市新区 135 个 DMA、7800 万元、年节水 1450 万 m³ → 5.38 元/(m³·a)",
        "来源性质": "CASE",
    },
    {
        "参数": "【官方锚点】净水工程单位水量投资估算指标",
        "取值低": 210.0, "取值高": 278.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "（交叉校验用，不可直接采用）",
        "来源": "《城市给水工程项目建设标准》建标 120-2009 表 7：建设规模 5–10 万m³/d 取 278–240，"
                "10–30 取 240–210，30–50 取 210–150 元/(m³·d)（规模越大取下限）",
        "来源性质": "OFFICIAL",
        "口径与风险备注": "① 表注明确说明按**北京市 1990 年价格**折算，**不能直接与当期造价比较**，"
                     "须用建安工程价格指数折算；② 指标为**新建**工程，不含场地准备、征地拆迁、"
                     "电贴费；③ 不含取水工程（表 6，57–90）与配水厂（表 8，55–95）。"
                     "三项合计约 322–463 元/(m³·d)（1990 价）。本文的 2170–2715 元/(m³·d) 是"
                     "**当期价的扩容增量**，与本锚点**口径与价格基准年都不同**，只作量级参照。",
    },
    {
        "参数": "【官方锚点】污水厂投资估算控制指标（一级A 新建）",
        "取值低": 1865.0, "取值高": 3825.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "（交叉校验用）",
        "来源": "《城市污水处理工程项目建设标准》建标 198-2022 表 5：新建一级A 常规污泥处理，"
                "I类 2175–1865、II类 2460–2175、III类 2810–2460、IV类 3825–2810 元/(m³·d)；"
                "准IV类水体提标：I类 1120–980、II类 1285–1120、III类 1530–1285、IV类 1965–1530",
        "来源性质": "OFFICIAL",
        "口径与风险备注": "表注明确：**按武汉市 2016 年 12 月人工/材料/机械预算价格计算**，"
                     "使用前须按当时当地价格调整；指标**不含征地拆迁、青苗与破路赔偿**，"
                     "且厂站主要设备按国产考虑。本文污水厂扩容增量取 1630–2170 元/(m³·d)，"
                     "**低于一级A 新建 I 类下限 1865**——因为扩容增量通常低于同规模新建，"
                     "但这一差异**未经证实**，是本文造价估计的**偏保守（偏低）方向**。",
    },
    {
        "参数": "【案例锚点】再生水回用单位投资（项目决算）",
        "取值低": 5595.0, "取值高": 5595.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "（交叉校验用）",
        "来源": "西安思源学院污水处理再生水回用工程（A²/O-MBR 4000 m³/d，含回用管网与泵站、"
                "含摊销）决算 2238.13 万元，单位投资 **5595.33 元/(m³·d)**"
                "（李东等，汉斯出版社《水污染及处理》2022）",
        "来源性质": "CASE",
        "口径与风险备注": "该值落在本文采用的 5000–7000 元/(m³·d) 区间内，是对 GREY 取值的正面印证。"
                     "但注意**规模效应**：4000 m³/d 属小型，本文情景为 4–6 万 m³/d，"
                     "规模大两个数量级，单位投资应更低——因此 5000–7000 对本文规模**可能偏高**。",
    },
    {
        "参数": "【口径冲突】中水回用单位投资（中央预算内投资估算标准）",
        "取值低": 1100.0, "取值高": 1100.0, "单位": "元/(m³·d⁻¹)",
        "适用干预": "（不作采用值）",
        "来源": "重点流域水环境综合治理专项中央预算内投资绩效目标表："
                "『污水提标改造和中水回用』估算标准 1100 万元/(万吨/日) = 1100 元/(m³·d)",
        "来源性质": "GREY",
        "口径与风险备注": "**与上面两个来源相差 5 倍**（1100 vs 5595–7000）。差异来自口径："
                     "该表是**中央预算内投资补助的估算标准**，很可能只覆盖部分工程内容或"
                     "仅指提标改造增量，而非完整再生水厂投资。"
                     "**本文不采用本值**，仅记录为第 5 例口径冲突（见 literature_benchmark_caliber.csv）。",
    },
    {
        "参数": "本地接水连接管线单位投资",
        "取值低": None, "取值高": None, "单位": "元/(m³·d⁻¹)",
        "适用干预": "local_connection_plus_*",
        "来源": "AUTHOR_INPUT_NEEDED：未检索到与 specific local_connection_capacity 口径对应的公开造价指标，暂不赋值",
        "来源性质": "MISSING",
    },
]

POWER_FACTOR = 0.95  # 110kV 用户站功率因数，用于 MW -> MVA
SIMULATION_YEARS = 2.0


def _unit_cost(name: str) -> tuple[float | None, float | None]:
    for row in COST_PARAMS:
        if row["参数"] == name:
            return row["取值低"], row["取值高"]
    raise KeyError(name)


def run_baseline_flows(project: dict, timeseries: pd.DataFrame) -> dict:
    """跑一次基线，取配水漏损量与居民未满足量（用于漏损干预的节水核算）。"""
    from aiuwm.full_engine import FullAIUWMModel

    result = FullAIUWMModel(project, timeseries).run()
    system = result.system_daily
    area = result.area_daily
    leakage_col = next((c for c in system.columns if "leakage" in c.lower()), None)
    return {
        "leakage_ml": float(system[leakage_col].sum()) if leakage_col else float("nan"),
        "domestic_unmet_ml": float(area["unmet_domestic_ml"].sum()) if not area.empty else 0.0,
    }


def intervention_cost(name: str, baseline: dict, specs: dict, domain: str,
                      saving_water_ml_year: float | None) -> dict:
    base = specs["baseline"]
    s = specs["interventions"]

    def find(n: str) -> list[dict]:
        return next((i["set"] for i in s if i["name"] == n), [])

    low = high = 0.0
    note: list[str] = []

    sets = find(name)

    for entry in sets:
        path, value = entry["path"], float(entry["value"])
        if path.startswith("components.WTW1.daily_capacity"):
            unit_l, unit_h = _unit_cost("给水厂扩容增量单位投资")
            delta_ml = value - float(base["wtw_daily_capacity_ml"])
            low += delta_ml * 1000.0 * unit_l / 1e4
            high += delta_ml * 1000.0 * unit_h / 1e4
            note.append(f"给水能力 +{delta_ml:.0f} ML/d")
        elif path.startswith("components.WWTW1.daily_capacity"):
            unit_l, unit_h = _unit_cost("污水处理厂扩容增量单位投资")
            delta_ml = value - float(base["wwtw_daily_capacity_ml"])
            low += delta_ml * 1000.0 * unit_l / 1e4
            high += delta_ml * 1000.0 * unit_h / 1e4
            note.append(f"污水处理 +{delta_ml:.0f} ML/d")
        elif path.startswith("components.CENTRAL_REUSE.treatment_capacity"):
            unit_l, unit_h = _unit_cost("中水回用单位投资")
            delta_ml = value - float(base["reuse_treatment_ml_day"])
            low += delta_ml * 1000.0 * unit_l / 1e4
            high += delta_ml * 1000.0 * unit_h / 1e4
            note.append(f"再生水处理能力 +{delta_ml:.0f} ML/d")
        elif path.startswith("components.AI_DC1.grid_connection_capacity"):
            low_u, high_u = _unit_cost("数据中心专用 110kV 变电站单位投资（案例区间）")
            delta_mw = value - float(base["grid_connection_mw"])
            delta_mva = delta_mw * 1000.0 / POWER_FACTOR
            low += delta_mva * low_u / 1e4
            high += delta_mva * high_u / 1e4
            note.append(f"接网容量 +{delta_mw:.0f} MW")
        elif path.startswith("components.AI_DC1.local_connection_capacity"):
            unit_l, unit_h = _unit_cost("本地接水连接管线单位投资")
            if unit_l is None:
                note.append("本地接水连接：造价指标缺失")
                return {"投资下限_万元": None, "投资上限_万元": None, "工程量说明": "; ".join(note)}
            low += 0.0
        elif path.startswith("components.DM1.leakage_fraction"):
            unit_l, unit_h = _unit_cost("漏损治理单位节水投资")
            if saving_water_ml_year is None:
                note.append("漏损治理：节水量未核算")
                return {"投资下限_万元": None, "投资上限_万元": None, "工程量说明": "; ".join(note)}
            low += saving_water_ml_year * 1000.0 * unit_l / 1e4
            high += saving_water_ml_year * 1000.0 * unit_h / 1e4
            note.append(f"年节水 {saving_water_ml_year:.0f} ML/a")

    return {
        "投资下限_万元": round(low, 1),
        "投资上限_万元": round(high, 1),
        "工程量说明": "; ".join(note),
    }


def dominates(a: dict, b: dict) -> bool:
    """a 支配 b：三目标（成本↓, 增益↑, 居民缺水↓）均不劣且至少一项更优。"""
    if a["投资中值_万元"] is None or b["投资中值_万元"] is None:
        return False
    cost_better = a["投资中值_万元"] <= b["投资中值_万元"]
    gain_better = a["CAWCC增益_MW"] >= b["CAWCC增益_MW"]
    unmet_better = a["居民未满足量增量_ML"] <= b["居民未满足量增量_ML"]
    all_not_worse = cost_better and gain_better and unmet_better
    strictly_better = (
        a["投资中值_万元"] < b["投资中值_万元"]
        or a["CAWCC增益_MW"] > b["CAWCC增益_MW"]
        or a["居民未满足量增量_ML"] < b["居民未满足量增量_ML"]
    )
    return all_not_worse and strictly_better


def main() -> None:
    rows: list[dict] = []

    # 成本参数表落盘（带来源，便于审稿溯源）
    params_frame = pd.DataFrame(COST_PARAMS)
    OUT.mkdir(parents=True, exist_ok=True)
    params_frame.to_csv(OUT / "cost_parameters.csv", index=False, encoding="utf-8-sig")

    for domain in ("ha", "co"):
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
        specs = json.loads(
            (R2 / "specs" / f"interventions_{domain}.json").read_text(encoding="utf-8")
        )
        marginal = pd.read_csv(OUT / domain / "intervention_marginal.csv")
        base_flows = run_baseline_flows(project, timeseries)

        # 居民未满足量：intervention_marginal.csv 里没有该列，必须在同一容量点实测。
        # 口径统一取 baseline 的 CAWCC（HA=1100 / CO=1400 MW），否则不同容量下不可比。
        from aiuwm.full_engine import FullAIUWMModel

        baseline_cawcc = float(
            marginal.loc[marginal["intervention"] == "baseline", "maximum_safe_ai_capacity_mw"].iloc[0]
        )
        unmet_by_intervention: dict[str, float] = {}
        for item in specs["interventions"]:
            trial = copy.deepcopy(project)
            for component in trial["components"].values():
                if component.get("kind") == "data_center":
                    component["installed_it_capacity_mw"] = baseline_cawcc
                    component.pop("capacity_schedule", None)
            for entry in item["set"]:
                target = trial
                parts = entry["path"].split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = entry["value"]
            area = FullAIUWMModel(trial, timeseries).run().area_daily
            unmet_by_intervention[item["name"]] = float(area["unmet_domestic_ml"].sum()) if not area.empty else 0.0
        baseline_unmet = unmet_by_intervention.get("baseline", 0.0)

        # 漏损干预的年节水量：用同一基线项目把 leakage_fraction 改到目标值再跑一次
        saving_ml_year = None
        leakage_spec = next(
            (i for i in specs["interventions"] if i["name"] == "leakage_minus_30pct"), None
        )
        if leakage_spec:
            target = float(leakage_spec["set"][0]["value"])
            trial = copy.deepcopy(project)
            trial["components"]["DM1"]["leakage_fraction"] = target
            result = FullAIUWMModel(trial, timeseries).run()
            system = result.system_daily
            leakage_col = next((c for c in system.columns if "leakage" in c.lower()), None)
            if leakage_col:
                reduced = float(system[leakage_col].sum())
                saving_ml_year = max((base_flows["leakage_ml"] - reduced) / SIMULATION_YEARS, 0.0)

        for _, record in marginal.iterrows():
            name = str(record["intervention"])
            cost = intervention_cost(name, base_flows, specs, domain, saving_ml_year)
            low, high = cost["投资下限_万元"], cost["投资上限_万元"]
            mid = (low + high) / 2.0 if (low is not None and high is not None) else None
            gain = float(record["capacity_gain_mw"])
            rows.append({
                "域": domain,
                "域标签": DOMAIN_LABEL[domain],
                "干预": name,
                "CAWCC增益_MW": gain,
                "干预后CAWCC_MW": float(record["maximum_safe_ai_capacity_mw"]),
                "限制约束": record["limiting_constraint"],
                "投资下限_万元": low,
                "投资上限_万元": high,
                "投资中值_万元": round(mid, 1) if mid is not None else None,
                "单位成本_万元每MW": (round(mid / gain, 2) if (mid and gain > 0) else None),
                "居民未满足量增量_ML": round(
                    unmet_by_intervention.get(name, 0.0) - baseline_unmet, 3
                ),
                "工程量说明": cost["工程量说明"],
            })

        print(f"[{domain}] 已处理 {len(rows)} 条累计（本域 "
              f"{len(marginal)} 个干预，baseline CAWCC={baseline_cawcc:.0f} MW）", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "intervention_cost_benefit.csv", index=False, encoding="utf-8-sig")

    # ---- Pareto 前沿（逐域） -------------------------------------------------
    pareto_rows = []
    for domain, group in frame.groupby("域"):
        # 候选集必须包含 baseline（投资=0，增益=0）。
        # 若把 baseline 排除，任何 0 增益但只要是最便宜的选项都会因为"没人能支配它"
        # 而误入 Pareto 有效集——这是纯粹的判定伪影（见 CO 的 leakage_minus_30pct）。
        candidates = group[group["投资中值_万元"].notna()].to_dict("records")
        for item in candidates:
            dominated = any(dominates(other, item) for other in candidates if other is not item)
            item["域"] = domain
            item["Pareto有效"] = not dominated
            pareto_rows.append(item)
    pareto_frame = pd.DataFrame(pareto_rows)
    pareto_frame.to_csv(OUT / "pareto_front.csv", index=False, encoding="utf-8-sig")

    # 把 Pareto 判定回填主表，使 intervention_cost_benefit.csv 自足可读
    if not pareto_frame.empty:
        flag = pareto_frame[["域", "干预", "Pareto有效"]].copy()
        frame = frame.merge(flag, on=["域", "干预"], how="left")
        frame["Pareto有效"] = frame["Pareto有效"].map(
            {True: "是", False: "否"}
        ).fillna("不适用（无投资额或未评估）")
        frame["零增益但仍有投资"] = (
            (frame["CAWCC增益_MW"] <= 0) & (frame["投资中值_万元"].fillna(0) > 0)
        ).map({True: "是", False: "否"})
        frame.to_csv(OUT / "intervention_cost_benefit.csv", index=False, encoding="utf-8-sig")

    print("=== 干预成本效益 ===")
    cols = ["域", "干预", "CAWCC增益_MW", "投资中值_万元", "单位成本_万元每MW"]
    if "Pareto有效" in frame.columns:
        cols.append("Pareto有效")
    print(frame[cols].to_string(index=False))
    print("\n=== Pareto 有效集 ===")
    if not pareto_frame.empty:
        print(pareto_frame[pareto_frame["Pareto有效"]][
            ["域", "干预", "CAWCC增益_MW", "投资中值_万元", "单位成本_万元每MW"]
        ].to_string(index=False))
    print(f"\n-> {OUT / 'intervention_cost_benefit.csv'}")
    print(f"-> {OUT / 'pareto_front.csv'}")
    print(f"-> {OUT / 'cost_parameters.csv'}")


if __name__ == "__main__":
    main()
