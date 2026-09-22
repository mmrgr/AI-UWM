"""把 R2 全部实验产物导出整理为可交付数据集。

产出三部分:
  1) csv/        按实验段分目录存放原始 CSV + 过程数据抽样
  2) Excel 分册  避开单个工作簿 1048576 行的硬上限
        册1 汇总册 : 索引 / 核心结果 / 各实验段聚合表 / 情景清单
        册2 过程册 : 关键表的逐日抽样(每 3 日一点) + 月/年聚合全量
        册3 组分册 : component_daily 抽样(表最宽，单独成册)
  3) README      数据说明、口径、文件索引、复现方式

逐日全量数据（约 156 万行 / 293 MB）不由本脚本复制，保留在
`aiuwm_src/validation_artifacts/timeseries/`，README 中给出路径与复现命令。
若确实需要把全量打包进交付目录，用 `--full-timeseries`。

用法:
    PYTHONPATH="src;scripts" python scripts/export_r2_dataset.py
    PYTHONPATH="src;scripts" python scripts/export_r2_dataset.py --out ../AI-UWM_R2_实验数据集
    PYTHONPATH="src;scripts" python scripts/export_r2_dataset.py --daily-step 7
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "validation_artifacts" / "r2"
TS = ROOT / "validation_artifacts" / "timeseries"

# 文件 -> (分类目录, 中文说明)
FILE_MAP: dict[str, tuple[str, str]] = {
    "realism_audit.csv": ("01_现实性审计", "19 项现实性判据审计结果（两域）"),
    "dry_first_calibration.csv": ("02_参数率定", "干式优先冷却切换阈值率定扫描"),
    "baseline_delta.csv": ("03_基线对比", "有/无 AI 数据中心的全期差值（剔除城市自然增长）"),
    "capacity_scan.csv": ("04_容量扫描", "0–2000 MW 容量扫描原始逐点结果"),
    "capacity_threshold.csv": ("04_容量扫描", "承载容量(CAWCC)判定结果与限制约束"),
    "constraint_boundaries.csv": ("04_容量扫描", "各约束的首次失效容量与裕度"),
    "constraint_informativeness.csv": ("04_容量扫描", "约束可失效性诊断（恒真判据识别）"),
    "intervention_marginal.csv": ("05_干预与边际", "各类干预方案对 CAWCC 的增益"),
    "reuse_marginal_curve.csv": ("05_干预与边际", "再生水「处理能力」杠杆边际曲线（无效杠杆）"),
    "reuse_production_marginal_curve.csv": ("05_干预与边际", "再生水「产水量」杠杆边际曲线（真杠杆）"),
    "intervention_cost_benefit.csv": ("05_干预与边际", "干预的投资区间与单位成本 万元/MW"),
    "pareto_front.csv": ("05_干预与边际", "投资-增益-居民缺水 三维 Pareto 判定"),
    "cost_parameters.csv": ("05_干预与边际", "造价参数表（含来源与来源性质）"),
    "reclaimed_coc_feedback.csv": ("06_机制实验", "再生水比例→水质约束 CoC→补水/排污 反馈链"),
    "reclaimed_net_substitution.csv": ("06_机制实验", "毛替代率 vs 净替代率（置换效应）"),
    "technology_comparison.csv": ("06_机制实验", "七种冷却技术对照（一技术一工程）"),
    "policy_comparison.csv": ("06_机制实验", "三种分配政策对照（居民未满足量）"),
    "state_pressure_matrix.csv": ("07_情景矩阵", "S×G 状态—压力 4×4 矩阵"),
    "heatwave_stress.csv": ("07_情景矩阵", "热浪极端情景（连续 15 天 +6 ℃）"),
    "morris.csv": ("08_不确定性", "Morris 筛选（AI 侧参数）"),
    "sobol.csv": ("08_不确定性", "Sobol 方差分解（AI 侧参数）"),
    "city_morris.csv": ("08_不确定性", "Morris 筛选（城市侧参数）"),
    "city_sobol.csv": ("08_不确定性", "Sobol 方差分解（城市侧参数，未收敛）"),
    "probabilistic_thresholds.csv": ("08_不确定性", "概率承载阈值原始样本（40 样本版）"),
    "probabilistic_summary.json": ("08_不确定性", "概率承载阈值 P05/P50/P95 汇总（40 样本版）"),
    "probabilistic_large.csv": ("08_不确定性", "概率承载阈值原始样本（1024 样本版）"),
    "probabilistic_large_summary.json": ("08_不确定性", "1024 样本分位数 + bootstrap 95% CI"),
    "probabilistic_sample_size_comparison.csv": ("08_不确定性", "40 vs 1024 样本的分位数估计对照"),
    "energy_rich_comparison.csv": ("09_能源富集", "能源富集补充情景与常规情景对比"),
    "city_gradient_capacity_scan.csv": ("10_城市梯度", "25 个合成城市 × 41 容量点全量扫描"),
    "city_gradient_cawcc.csv": ("10_城市梯度", "每个合成城市的 CAWCC 与限制约束"),
    "city_gradient_bottleneck.csv": ("10_城市梯度", "每个合成城市的瓶颈迁移链"),
    "literature_benchmark.csv": ("11_文献对标", "模型 WUE/PUE vs 披露值与行业基准"),
    "literature_benchmark_sources.csv": ("11_文献对标", "对标值来源清单与来源性质"),
    "literature_benchmark_model.csv": ("11_文献对标", "参与对标的模型侧取值"),
    "literature_benchmark_city.csv": ("11_文献对标", "城市侧指标 vs 官方标准与政策锚点（含判定）"),
    "literature_benchmark_city_sources.csv": ("11_文献对标", "城市侧对标锚点（含比较方式与对标口径）"),
    "literature_benchmark_city_model.csv": ("11_文献对标", "模型城市侧取值（同一指标多口径并列）"),
    "literature_benchmark_caliber.csv": ("11_文献对标", "口径冲突警示 4 例（跨文献比较前必读）"),
    "r2_verification.json": ("12_自检", "V1–V12 机器自检结果"),
}

DOMAIN_LABEL = {"ha": "D-HA 缺水—高冷负荷端", "co": "D-CO 气候凉爽端"}


def collect_artifacts() -> list[tuple[Path, str, str]]:
    rows: list[tuple[Path, str, str]] = []
    for name, (category, desc) in FILE_MAP.items():
        for domain in ("ha", "co"):
            path = SRC / domain / name
            if path.exists():
                rows.append((path, f"{category}/{domain}", f"[{domain.upper()}] {desc}"))
        top = SRC / name
        if top.exists():
            rows.append((top, category, desc))
    return rows


def safe_sheet(name: str) -> str:
    cleaned = "".join(c for c in name if c not in "[]:*?/\\")
    return cleaned[:31]


def build_key_results() -> pd.DataFrame:
    records: list[dict] = []

    def add(domain: str, category: str, metric: str, value, unit: str, note: str = "") -> None:
        records.append({
            "域": DOMAIN_LABEL.get(domain, domain),
            "类别": category,
            "指标": metric,
            "值": value,
            "单位": unit,
            "说明": note,
        })

    ver = SRC / "r2_verification.json"
    for domain in ("ha", "co"):
        folder = SRC / domain

        if ver.exists():
            vd = json.loads(ver.read_text(encoding="utf-8")).get(domain, {})
            for key, label in (
                ("cawcc_annual", "CAWCC(年尺度)"),
                ("cawcc_daily", "CAWCC(日尺度)"),
                ("cawcc_hourly", "CAWCC(小时尺度)"),
            ):
                blk = vd.get(key) or {}
                add(domain, "承载容量", label, blk.get("maximum_safe_ai_capacity_mw"), "MW",
                    f"限制约束: {blk.get('limiting_constraint')}")

        inter = folder / "intervention_marginal.csv"
        if inter.exists():
            df = pd.read_csv(inter)
            cols = {c.lower(): c for c in df.columns}
            cap, name = cols.get("maximum_safe_ai_capacity_mw"), cols.get("intervention")
            gain = cols.get("capacity_gain_mw")
            if cap and name:
                for _, r in df.iterrows():
                    note = f"相对同网格 baseline 增益 {r[gain]:+.0f} MW" if gain else ""
                    add(domain, "干预增益", str(r[name]), r[cap], "MW", note)

        psum = folder / "probabilistic_large_summary.json"
        if psum.exists():
            d = json.loads(psum.read_text(encoding="utf-8"))
            for k in ("p05", "p50", "p95"):
                ci = d.get(f"{k}_ci95") or [None, None]
                add(domain, "概率承载", f"CAWCC {k.upper()} (1024 样本)", d.get(k), "MW",
                    f"bootstrap 95%CI [{ci[0]}, {ci[1]}]; {d.get('samples')} 样本")
            exc = d.get("exceedance_probability_ci95") or [None, None]
            add(domain, "概率承载", "1000 MW 超限概率 (1024 样本)", d.get("exceedance_probability"), "-",
                f"bootstrap 95%CI [{exc[0]}, {exc[1]}]")

        cm = folder / "city_morris.csv"
        if cm.exists():
            df = pd.read_csv(cm)
            for metric, sub in df.groupby("metric"):
                top = sub.sort_values("mu_star", ascending=False).iloc[0]
                add(domain, "城市侧敏感性", f"Morris 首位参数({metric})",
                    top.get("parameter"), "-", f"mu* = {top.get('mu_star'):.4g}")

        er = folder / "energy_rich_comparison.csv"
        if er.exists():
            df = pd.read_csv(er, encoding="utf-8-sig")
            for _, r in df.iterrows():
                add(domain, "能源富集", f"{r['情景']} CAWCC(日)", r.get("CAWCC_日_MW"), "MW",
                    f"限制约束: {r.get('日尺度限制约束')}")

        sg = folder / "state_pressure_matrix.csv"
        if sg.exists():
            df = pd.read_csv(sg, encoding="utf-8-sig")
            rel = [c for c in df.columns if "reliability" in c.lower()]
            if rel:
                c = rel[0]
                worst = df.loc[df[c].idxmin()]
                add(domain, "情景矩阵", "S×G 最不利状态（按系统可靠性）",
                    worst.get("state") or worst.iloc[0], "-", f"{c} = {worst[c]:.3f}")

        hw = folder / "heatwave_stress.csv"
        if hw.exists():
            df = pd.read_csv(hw, encoding="utf-8-sig")
            cand = [c for c in df.columns if "ratio" in c.lower() or "utilization" in c.lower()]
            if cand:
                c = cand[0]
                add(domain, "热浪", f"热浪下最大 {c}", df[c].max(), "-",
                    f"{len(df)} 个容量点，判据阈值 0.95")

        cb = SRC / "intervention_cost_benefit.csv"
        if cb.exists():
            df = pd.read_csv(cb, encoding="utf-8-sig")
            sub = df[df["域"] == domain]
            affordable = sub[(sub["单位成本_万元每MW"].notna()) & (sub["CAWCC增益_MW"] > 0)]
            if not affordable.empty:
                best = affordable.loc[affordable["单位成本_万元每MW"].idxmin()]
                add(domain, "成本效益", "单位成本最优干预", best["干预"], "万元/MW",
                    f"{best['单位成本_万元每MW']:.1f} 万元/MW，投资中值 "
                    f"{best['投资中值_万元']:.0f} 万元，增益 {best['CAWCC增益_MW']:.0f} MW")
            # 基线瓶颈约束：决定"哪一类干预才对症"
            base_row = sub[sub["干预"] == "baseline"]
            if not base_row.empty:
                add(domain, "瓶颈分工", "基线限制约束", base_row.iloc[0]["限制约束"], "-",
                    "决定扩容瓶颈在物理水侧还是电网侧")
            wasted = sub[sub["零增益但仍有投资"] == "是"]
            if not wasted.empty:
                add(domain, "瓶颈分工", "零增益但仍需投资的干预数", len(wasted), "个",
                    f"合计投资中值 {wasted['投资中值_万元'].sum():,.0f} 万元，"
                    f"CAWCC 增益均为 0 MW")

        # ---- 城市侧外部效度：模型指标 vs 官方标准/政策锚点 --------------------
        city_cmp = SRC / "literature_benchmark_city.csv"
        if city_cmp.exists():
            df = pd.read_csv(city_cmp, encoding="utf-8-sig")
            sub = df[df["模型来源"].str.endswith(domain.upper())]
            for metric, label in (
                ("综合漏损率_pct", "综合漏损率"),
                ("人均综合生活用水_L_per_cap_d", "人均生活用水"),
                ("再生水利用率_pct", "再生水利用率"),
            ):
                block = sub[sub["指标"] == metric]
                for _, r in block.iterrows():
                    add(domain, "城市侧外部效度",
                        f"{label}｜{r['模型口径']}", r["模型值"],
                        "L/(人·d)" if "人均" in metric else "pct",
                        f"vs {r['对标主体']}：{r['判定']}（偏差 {r['偏差']:+.2f}）")

    return pd.DataFrame(records)


def sample_timeseries(out: Path, step: int, index_rows: list[dict]) -> dict[str, pd.DataFrame]:
    """对逐日过程数据做等距抽样，并把月聚合全量拼成一张宽表。"""
    manifest_path = TS / "manifest.csv"
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")

    sampled_core: dict[str, list[pd.DataFrame]] = {t: [] for t in ("system", "area", "data_center")}
    monthly_all: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    exported = 0

    ts_dir = out / "csv" / "13_逐日过程_抽样"
    ts_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        folder = TS / str(row["路径"])
        if not folder.exists():
            continue
        label = f"{row['域']}/{row['情景']}"
        exported += 1

        for table in ("system", "area", "data_center"):
            daily = folder / f"{table}_daily.csv"
            if not daily.exists():
                continue
            try:
                frame = pd.read_csv(daily, encoding="utf-8-sig")
            except Exception:
                continue
            frame.insert(0, "情景", label)
            sampled_core[table].append(frame.iloc[::step])

        monthly = folder / "system_monthly.csv"
        if monthly.exists():
            try:
                frame = pd.read_csv(monthly, encoding="utf-8-sig")
                frame.insert(0, "情景", label)
                monthly_all.append(frame)
            except Exception:
                pass

        daily = folder / "component_daily.csv"
        if daily.exists():
            try:
                frame = pd.read_csv(daily, encoding="utf-8-sig")
                frame.insert(0, "情景", label)
                component_parts.append(frame.iloc[::step])
            except Exception:
                pass

    result: dict[str, pd.DataFrame] = {}

    def record(name: str, file_name: str, note: str) -> None:
        size = (ts_dir / file_name).stat().st_size
        index_rows.append({
            "分类": "13_逐日过程_抽样",
            "文件": file_name,
            "相对路径": f"csv/13_逐日过程_抽样/{file_name}",
            "说明": note,
            "大小_字节": size,
        })

    for table, frames in sampled_core.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        file_name = f"{table}_daily_sampled.csv"
        combined.to_csv(ts_dir / file_name, index=False, encoding="utf-8-sig")
        result[f"{table}_daily_sampled"] = combined
        record(table, file_name, f"{len(combined):,} 行，逐日每 {step} 日取一点（全量见仓库）")

    if monthly_all:
        combined = pd.concat(monthly_all, ignore_index=True)
        combined.to_csv(ts_dir / "system_monthly_all.csv", index=False, encoding="utf-8-sig")
        result["system_monthly_all"] = combined
        record("monthly", "system_monthly_all.csv", f"{len(combined):,} 行，全情景月聚合（全量）")

    if component_parts:
        combined = pd.concat(component_parts, ignore_index=True)
        combined.to_csv(ts_dir / "component_daily_sampled.csv", index=False, encoding="utf-8-sig")
        result["_component"] = combined
        record("component", "component_daily_sampled.csv",
               f"{len(combined):,} 行，组件级逐日抽样（单独成册）")

    size = manifest_path.stat().st_size
    index_rows.append({
        "分类": "13_逐日过程_抽样",
        "文件": "manifest.csv",
        "相对路径": "csv/13_逐日过程_抽样/manifest.csv",
        "说明": f"{len(manifest):,} 个情景的元数据与关键 KPI（全量）",
        "大小_字节": size,
    })
    shutil.copy2(manifest_path, ts_dir / "manifest.csv")
    print(f"  抽样 {exported} 个情景, step={step}")
    return result


def write_readme(out: Path, index_rows: list[dict], key: pd.DataFrame, step: int,
                 sampled: dict[str, pd.DataFrame]) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# AI-UWM R2 实验数据集",
        "",
        f"导出时间：{stamp}",
        "来源：`aiuwm_src/validation_artifacts/`（`r2/` 聚合结果 + `timeseries/` 逐日过程）",
        "对应方案：《AI-UWM 下一轮模拟：情景设置与实施方案》v4",
        "",
        "## 交付结构",
        "",
        "```",
        "AI-UWM_R2_实验数据集/",
        "├── README_数据说明.md          本文件",
        "├── AI-UWM_R2_册1_汇总.xlsx    索引 / 核心结果 / 各实验段聚合表 / 情景清单",
        "├── AI-UWM_R2_册2_过程.xlsx    逐日抽样 + 全情景月聚合",
        "├── AI-UWM_R2_册3_组分.xlsx    组件级逐日抽样（表最宽，单独成册）",
        "└── csv/                      按实验段分目录的 CSV 源文件",
        "```",
        "",
        "分册原因：单个 Excel 工作簿有 1,048,576 行上限，逐日全量写 xlsx 会产生 GB 级文件。",
        f"逐日抽样步长为 **每 {step} 日取一点**；月/年聚合为全量。",
        "",
        "## 数据规模",
        "",
        "- 逐日过程全量：45 个情景 / 1,563,174 行 / 293 MB",
        "  （保留在仓库 `aiuwm_src/validation_artifacts/timeseries/`，未随本数据集复制）",
    ]
    for name, frame in sampled.items():
        if not name.startswith("_"):
            lines.append(f"- 抽样表 `{name}`：{len(frame):,} 行")
    lines += [
        f"- 核心结果长表：{len(key)} 条",
        "",
        "## 口径提醒（引用数据时必须遵守）",
        "",
        "1. 模拟期 **730 天（2 年）**；引擎 `*_ml` 字段为**全期累计**，`*_ml_d` 为**日均值**。",
        "   与文献年值比较必须先年化（÷2）。",
        "2. 容量扫描网格 0–2000 MW 是**刻意设定的压力测试上界**。",
        "   现实百万人口城市内的园区为 50–300 MW，**结论引用前必须折算到该区间**。",
        "3. 驱动为**合成序列**，任何数字都不得表述为真实城市的实测结果。",
        "4. 规划裕度类判据阈值为 **0.95**（非 1.00）——引擎对能力做硬截断，利用率恒 ≤1。",
        "5. 绝对量报 3 位有效数字，无量纲比值保留 3 位小数（见方案 1.5 节）。",
        "6. **城市梯度的中间城市是两端点的线性混合**，是合成谱而非真实城市，",
        "   仅用于检验结论沿城市参数轴的连续性与单调性。",
        "7. 造价与对标数据的**来源性质已分级标注**（见 `05_干预与边际/cost_parameters.csv` 与",
        "   `11_文献对标/literature_benchmark_sources.csv`）：",
        "   OFFICIAL（政府或电网正式文件）/ CASE（具体项目批复或中标）/",
        "   DISCLOSURE（公司披露转述）/ INDUSTRY（行业基准）/ GREY（二手摘录或行业网页）。",
        "   **除 OFFICIAL 与 CASE 外，其余条目论文正式引用前必须回溯到原始年报、ESG 报告或造价定额。**",
        "8. **跨文献比较前先看口径**：`11_文献对标/literature_benchmark_caliber.csv` 记录了 4 例",
        "   同一指标在不同来源/不同定义下不可直接比较的实例（WUE 同一公司同一年相差近 1 倍；",
        "   城市漏损率同一年官方口径相差 >2.8 个百分点；再生水『利用率』与『替代率』分母不同；",
        "   毛替代率 vs 净替代率）。**引用时必须写明口径，不得跨口径混用数值。**",
        "9. 城市侧对标使用 `cap_0000`（无 AI 负荷）作为城市基线，避免 AI 负荷污染城市指标。",
        "   人均用水已区分**平均日/最高日**两个口径，只与 GB 50013-2018 同口径定额比对。",
        "10. Pareto 候选集**包含 baseline（投资 0、增益 0）**。若不纳入，零增益但最便宜的选项",
        "    会因无人能支配它而误入有效集——这是判定伪影，不是结果。",
        "",
        "## 文件索引",
        "",
        "| 分类 | 文件 | 说明 |",
        "|---|---|---|",
    ]
    lines += [f"| {r['分类']} | `{r['文件']}` | {r['说明']} |" for r in index_rows]
    lines += [
        "",
        "## 复现方式",
        "",
        "```bash",
        "cd aiuwm_src",
        'export PYTHONPATH="src;scripts"',
        "python scripts/build_r2_domains.py --years 2 --seed 20260918   # 建域",
        "python scripts/audit_realism.py --domains ha,co --ai-capacity 1000        # 审计(须 0 FAIL)",
        "python scripts/calibrate_dry_first.py --domains ha,co --capacity 750",
        "python scripts/run_r2.py --domains ha,co                      # 主矩阵",
        "python scripts/run_r2_sensitivity.py --domains ha,co          # AI 侧敏感性",
        "python scripts/run_r2_extra.py --domains ha,co                # 城市侧敏感性 + 能源富集",
        "python scripts/run_r2_probabilistic.py --samples 1024 --workers 24   # 大样本概率阈值",
        "python scripts/run_r2_city_gradient.py --grid 5 --workers 16  # 25 城梯度",
        "python scripts/run_r2_cost_pareto.py                          # 成本效益与 Pareto",
        "python scripts/build_literature_benchmark.py                  # 文献对标",
        "python scripts/run_r2_timeseries.py --domains ha,co           # 逐日过程全量落盘",
        "python scripts/export_r2_dataset.py                           # 本数据集",
        "```",
        "",
        "种子固定为 `20260918 + 101 × 域序号`，不使用 `hash(str)`（避免 PYTHONHASHSEED 不确定性）。",
    ]
    (out / "README_数据说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT.parent / "AI-UWM_R2_实验数据集"))
    parser.add_argument("--daily-step", type=int, default=3, help="逐日抽样步长")
    parser.add_argument("--full-timeseries", action="store_true", help="同时复制逐日全量")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.xlsx"):
        stale.unlink()

    rows = collect_artifacts()
    print(f"采集 {len(rows)} 个产物文件")

    index_rows: list[dict] = []
    for src, category, desc in rows:
        dest_dir = out / "csv" / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)
        index_rows.append({
            "分类": category,
            "文件": src.name,
            "相对路径": str((dest_dir / src.name).relative_to(out)).replace("\\", "/"),
            "说明": desc,
            "大小_字节": src.stat().st_size,
        })

    sampled = sample_timeseries(out, args.daily_step, index_rows)

    if args.full_timeseries and TS.exists():
        dest = out / "timeseries_full"
        print("  复制逐日全量 ...")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(TS, dest)

    key = build_key_results()
    print(f"核心结果 {len(key)} 条")

    # ---- 册1 汇总 -----------------------------------------------------------
    book1 = out / "AI-UWM_R2_册1_汇总.xlsx"
    used: set[str] = set()
    with pd.ExcelWriter(book1, engine="openpyxl") as writer:
        pd.DataFrame(index_rows).to_excel(writer, sheet_name="00_文件索引", index=False)
        used.add("00_文件索引")
        key.to_excel(writer, sheet_name="01_核心结果", index=False)
        used.add("01_核心结果")
        manifest_path = TS / "manifest.csv"
        if manifest_path.exists():
            pd.read_csv(manifest_path, encoding="utf-8-sig").to_excel(
                writer, sheet_name="02_情景清单", index=False
            )
            used.add("02_情景清单")
        for src, category, desc in rows:
            base = f"{category.replace('/', '_')}_{src.stem}"
            sheet = safe_sheet(base)
            n = 2
            while sheet in used:
                sheet = safe_sheet(f"{base}_{n}")
                n += 1
            used.add(sheet)
            try:
                if src.suffix == ".json":
                    try:
                        df = pd.json_normalize(json.loads(src.read_text(encoding="utf-8")))
                    except Exception:
                        df = pd.DataFrame({"raw": src.read_text(encoding="utf-8").splitlines()})
                else:
                    df = pd.read_csv(src, encoding="utf-8-sig")
                df.to_excel(writer, sheet_name=sheet, index=False)
            except Exception as exc:
                print(f"  skip {src.name}: {exc}")
    print(f"册1 -> {book1}")

    # ---- 册2 过程（核心表抽样 + 月聚合） ------------------------------------
    core = {k: v for k, v in sampled.items() if not k.startswith("_")}
    if core:
        book2 = out / "AI-UWM_R2_册2_过程.xlsx"
        with pd.ExcelWriter(book2, engine="openpyxl") as writer:
            for name, frame in core.items():
                frame.to_excel(writer, sheet_name=safe_sheet(name), index=False)
        print(f"册2 -> {book2}")

    # ---- 册3 组分（最宽的表单独成册） ---------------------------------------
    if "_component" in sampled:
        book3 = out / "AI-UWM_R2_册3_组分.xlsx"
        with pd.ExcelWriter(book3, engine="openpyxl") as writer:
            sampled["_component"].to_excel(writer, sheet_name="component_sampled", index=False)
        print(f"册3 -> {book3}")

    write_readme(out, index_rows, key, args.daily_step, sampled)
    print(f"\n导出完成 -> {out}")


if __name__ == "__main__":
    main()
