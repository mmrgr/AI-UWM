"""把 R2 全部实验产物导出整理为可交付数据集。

产出三部分:
  1) CSV 目录     —— 按实验段分目录存放原始 CSV(程序可读, 保持原样)
  2) Excel 工作簿 —— 每张表一个 sheet + 索引 sheet(人工可看)
  3) README       —— 数据说明、口径、文件索引

用法:
    python scripts/export_r2_dataset.py
    python scripts/export_r2_dataset.py --out ../AI-UWM_R2_数据集
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
    "reclaimed_coc_feedback.csv": ("06_机制实验", "再生水比例→水质约束 CoC→补水/排污 反馈链"),
    "reclaimed_net_substitution.csv": ("06_机制实验", "毛替代率 vs 净替代率（置换效应）"),
    "technology_comparison.csv": ("06_机制实验", "七种冷却技术对照（一技术一工程）"),
    "policy_comparison.csv": ("06_机制实验", "三种分配政策对照（居民未满足量）"),
    "state_pressure_matrix.csv": ("07_情景矩阵", "S×G 状态—压力 4×4 矩阵"),
    "heatwave_stress.csv": ("07_情景矩阵", "热浪极端情景（连续 15 天 +6 ℃）"),
    "morris.csv": ("08_不确定性", "Morris 筛选（AI 侧参数）"),
    "sobol.csv": ("08_不确定性", "Sobol 方差分解（AI 侧参数）"),
    "city_morris.csv": ("08_不确定性", "Morris 筛选（城市侧参数，v4 新增）"),
    "city_sobol.csv": ("08_不确定性", "Sobol 方差分解（城市侧参数，v4 新增）"),
    "probabilistic_thresholds.csv": ("08_不确定性", "概率承载阈值原始样本"),
    "probabilistic_summary.json": ("08_不确定性", "概率承载阈值 P05/P50/P95 汇总"),
    "energy_rich_comparison.csv": ("09_能源富集", "能源富集补充情景与常规情景对比（v4 新增）"),
    "r2_verification.json": ("10_自检", "V1–V12 机器自检结果"),
}

DOMAIN_LABEL = {"ha": "D-HA 缺水—高冷负荷端", "co": "D-CO 气候凉爽端"}


def collect() -> list[tuple[Path, str, str]]:
    """收集所有产物文件 -> [(源路径, 分类目录, 说明)]"""
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
    """Excel sheet 名限制: <=31 字符, 不含 []:*?/\\"""
    cleaned = "".join(c for c in name if c not in "[]:*?/\\")
    return cleaned[:31]


def build_key_results() -> pd.DataFrame:
    """从产物中自动提取核心结论指标，生成长表。"""
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

    for domain in ("ha", "co"):
        folder = SRC / domain

        # CAWCC 与限制约束（取自自检汇总，键名 maximum_safe_ai_capacity_mw）
        ver = SRC / "r2_verification.json"
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

        # 干预增益
        inter = folder / "intervention_marginal.csv"
        if inter.exists():
            df = pd.read_csv(inter)
            cols = {c.lower(): c for c in df.columns}
            cap = cols.get("maximum_safe_ai_capacity_mw")
            name = cols.get("intervention")
            gain = cols.get("capacity_gain_mw")
            if cap and name:
                for _, r in df.iterrows():
                    note = f"相对同网格 baseline 增益 {r[gain]:+.0f} MW" if gain else ""
                    add(domain, "干预增益", str(r[name]), r[cap], "MW", note)

        # 概率阈值
        psum = folder / "probabilistic_summary.json"
        if psum.exists():
            d = json.loads(psum.read_text(encoding="utf-8"))
            for k in ("p05", "p50", "p95"):
                add(domain, "概率承载", f"CAWCC {k.upper()}", d.get(k), "MW",
                    f"{d.get('samples')} 样本")
            add(domain, "概率承载", "1000 MW 超限概率", d.get("exceedance_probability"), "-",
                f"拟建 {d.get('proposed_mw')} MW")

        # 净替代率
        net = folder / "reclaimed_net_substitution.csv"
        if net.exists():
            df = pd.read_csv(net)
            for _, r in df.iterrows():
                label = None
                for c in df.columns:
                    if c.lower() in ("tag", "label", "config", "scenario"):
                        v = r[c]
                        if isinstance(v, str) and v.strip():
                            label = v.strip()
                            break
                if label is None:
                    label = "容量扫描"
                if "capacity_mw" in df.columns:
                    label = f"{label} @ {r['capacity_mw']:.0f}MW"
                for c in df.columns:
                    if "net" in c.lower() and "substitution" in c.lower():
                        add(domain, "替代率", f"净替代率({label})", r[c], "-", "毛替代率对照见同表")
                    if "gross" in c.lower() or ("substitution_ratio" in c.lower() and "net" not in c.lower()):
                        add(domain, "替代率", f"毛替代率({label})", r[c], "-", "")

        # 城市侧敏感性（v4）
        cm = folder / "city_morris.csv"
        if cm.exists():
            df = pd.read_csv(cm)
            for metric, sub in df.groupby("metric"):
                sub = sub.sort_values("mu_star", ascending=False)
                top = sub.iloc[0]
                add(domain, "城市侧敏感性", f"Morris 首位参数({metric})",
                    top.get("parameter"), "-", f"mu* = {top.get('mu_star'):.4g}")

        # 能源富集（v4：两端点）
        er = folder / "energy_rich_comparison.csv"
        if er.exists():
            df = pd.read_csv(er, encoding="utf-8-sig")
            for _, r in df.iterrows():
                add(domain, "能源富集", f"{r['情景']} CAWCC(日)", r.get("CAWCC_日_MW"), "MW",
                    f"限制约束: {r.get('日尺度限制约束')}")

        # S×G 状态—压力矩阵：最不利状态
        sg = folder / "state_pressure_matrix.csv"
        if sg.exists():
            df = pd.read_csv(sg, encoding="utf-8-sig")
            rel = [c for c in df.columns if "reliability" in c.lower()]
            if rel:
                c = rel[0]
                worst = df.loc[df[c].idxmin()]
                state = worst.get("state") or worst.get("state_id") or worst.iloc[0]
                add(domain, "情景矩阵", "S×G 最不利状态（按系统可靠性）", state, "-",
                    f"{c} = {worst[c]:.3f}")

        # 热浪：最大接入比
        hw = folder / "heatwave_stress.csv"
        if hw.exists():
            df = pd.read_csv(hw, encoding="utf-8-sig")
            cand = [c for c in df.columns if "ratio" in c.lower() or "utilization" in c.lower()]
            if cand:
                c = cand[0]
                add(domain, "热浪", f"热浪下最大 {c}", df[c].max(), "-",
                    f"{len(df)} 个容量点，判据阈值 0.95")

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT.parent / "AI-UWM_R2_实验数据集"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 幂等：仅删除旧的汇总工作簿，CSV 目录直接覆盖（不做 rmtree，避免误删用户文件）
    stale_xlsx = out / "AI-UWM_R2_实验数据汇总.xlsx"
    if stale_xlsx.exists():
        stale_xlsx.unlink()

    rows = collect()
    print(f"collected {len(rows)} artifact files")

    # ---- 1) CSV 目录 ------------------------------------------------------
    index_rows = []
    for src, category, desc in rows:
        dest_dir = out / "csv" / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        index_rows.append({
            "分类": category,
            "文件": src.name,
            "相对路径": str(dest.relative_to(out)).replace("\\", "/"),
            "说明": desc,
            "大小_字节": src.stat().st_size,
        })

    # ---- 2) Excel 工作簿 --------------------------------------------------
    xlsx = out / "AI-UWM_R2_实验数据汇总.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(index_rows).to_excel(writer, sheet_name="00_索引", index=False)

        key = build_key_results()
        key.to_excel(writer, sheet_name="01_核心结果", index=False)

        used: set[str] = {"00_索引", "01_核心结果"}
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
                    text = src.read_text(encoding="utf-8")
                    try:
                        df = pd.json_normalize(json.loads(text))
                    except Exception:
                        df = pd.DataFrame({"raw": text.splitlines()})
                else:
                    df = pd.read_csv(src, encoding="utf-8-sig")
                df.to_excel(writer, sheet_name=sheet, index=False)
            except Exception as exc:  # pragma: no cover
                print(f"  skip {src.name}: {exc}")

    # ---- 3) README --------------------------------------------------------
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# AI-UWM R2 实验数据集",
        "",
        f"导出时间：{stamp}",
        f"来源：`aiuwm_src/validation_artifacts/r2/`",
        f"对应方案：《AI-UWM 下一轮模拟：情景设置与实施方案》v4",
        "",
        "## 目录结构",
        "",
        "```",
        "AI-UWM_R2_实验数据集/",
        "├── README_数据说明.md        本文件",
        "├── AI-UWM_R2_实验数据汇总.xlsx  全部表格（一表一 sheet，含 00_索引 与 01_核心结果）",
        "└── csv/                      按实验段分目录的原始 CSV",
        "```",
        "",
        "## 口径提醒（引用数据时必须遵守）",
        "",
        "1. 模拟期 **730 天（2 年）**；引擎 `*_ml` 字段为**全期累计**，`*_ml_d` 为**日均值**。",
        "   与文献年值比较必须先年化（÷2）。",
        "2. 容量扫描网格 0–2000 MW 是**刻意设定的压力测试上界**，用于定位瓶颈迁移点。",
        "   现实百万人口城市内的园区为 50–300 MW，**结论引用前必须折算到该区间**。",
        "3. 驱动为**合成序列**（`driver_series: synthetic`），",
        "   任何数字都不得表述为真实城市的实测结果。",
        "4. 规划裕度类判据阈值为 **0.95**（非 1.00）——引擎对能力做硬截断，利用率恒 ≤1，`≤1` 恒真。",
        "5. 绝对量报 3 位有效数字，无量纲比值判据保留 3 位小数（见方案 1.5 节）。",
        "",
        "## 文件索引",
        "",
        "| 分类 | 文件 | 说明 |",
        "|---|---|---|",
    ]
    for r in index_rows:
        lines.append(f"| {r['分类']} | `{r['文件']}` | {r['说明']} |")

    lines += [
        "",
        "## 复现方式",
        "",
        "```bash",
        "cd aiuwm_src",
        "export PYTHONPATH=src",
        "python scripts/build_r2_domains.py --years 2 --seed 20260918   # 建域",
        "python scripts/audit_realism.py --domains ha,co --ai-capacity 1000  # 审计(须 0 FAIL)",
        "python scripts/calibrate_dry_first.py --domains ha,co --capacity 750",
        "python scripts/run_r2.py --domains ha,co",
        'PYTHONPATH="src;scripts" python scripts/run_r2_extra.py --domains ha,co',
        "python scripts/export_r2_dataset.py",
        "```",
        "",
        "种子固定为 `20260918 + 101 × 域序号`，不使用 `hash(str)`（避免 PYTHONHASHSEED 不确定性）。",
    ]
    (out / "README_数据说明.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n导出完成 -> {out}")
    print(f"  CSV 目录: {out / 'csv'}")
    print(f"  Excel   : {xlsx}")
    print(f"  核心结果: {len(build_key_results())} 条")


if __name__ == "__main__":
    main()
