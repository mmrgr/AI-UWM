"""R2 逐日过程数据全量落盘（任务 A）。

之前的所有实验脚本只提取了 .max()/.mean() 之类的聚合标量，把引擎本来就
产出的 13 张逐日过程表全部丢弃了。本脚本把它们完整落盘，并提供日/周/月/年
四级聚合。

情景集合（每域）:
  Tier1 全表 : cap_0000 ... cap_2000 共 9 个容量点   -> FullModelResult.write()
  Tier2 精简 : tech_* / policy_* / g{G1..G3} / energy_rich -> 4 张核心表

产出:
  validation_artifacts/timeseries/{domain}/{scenario}/<表>.csv
  validation_artifacts/timeseries/manifest.csv   情景元数据 + 关键 KPI

用法:
    python scripts/run_r2_timeseries.py
    python scripts/run_r2_timeseries.py --domains ha --tier 1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aiuwm.ai_metrics import summarize_ai_water_kpis  # noqa: E402
from aiuwm.full_engine import FullAIUWMModel, _aggregate_frame  # noqa: E402

R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "timeseries"

CAPACITIES = [0.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0]

DOMAIN_LABEL = {"ha": "D-HA 缺水—高冷负荷端", "co": "D-CO 气候凉爽端"}

# Tier2 仅保留这四张核心表（学术分析最常用，且与约束判据直接对应）
SLIM_TABLES = {
    "system": (None, []),
    "component": (None, ["component_id", "kind"]),
    "area": (None, ["area_id"]),
    "data_center": (None, ["data_center_id"]),
}


def _set_capacity(project: dict, capacity_mw: float) -> dict:
    trial = copy.deepcopy(project)
    for component in trial["components"].values():
        if component.get("kind") == "data_center":
            component["installed_it_capacity_mw"] = float(capacity_mw)
            component.pop("capacity_schedule", None)
    return trial


def _apply_g(project: dict, timeseries: pd.DataFrame, pressure: str) -> tuple[dict, pd.DataFrame]:
    """外部压力层 G1/G2/G3（与 run_r2.apply_g 同口径）。"""
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
                component["abstraction_capacity_ml_day"] = (
                    float(component["abstraction_capacity_ml_day"]) * 0.85
                )
        if pressure == "G2":
            drivers.loc[summer, "temperature_c"] = drivers.loc[summer, "temperature_c"].astype(float) + 1.0
    if pressure == "G3":
        for area in trial["local_areas"].values():
            for profile in area.get("demand_profiles", []):
                if "base_value" in profile:
                    profile["base_value"] = float(profile["base_value"]) * 1.12
    return trial, drivers


def write_full(result, dest: Path) -> tuple[int, int]:
    """Tier1: 全部 13 张表 + 周/月/年聚合。返回 (行数, 字节数)。"""
    dest.mkdir(parents=True, exist_ok=True)
    result.write(dest)
    rows = 0
    size = 0
    for p in dest.rglob("*.csv"):
        size += p.stat().st_size
        rows += max(sum(1 for _ in p.open(encoding="utf-8")) - 1, 0)
    return rows, size


def write_slim(result, dest: Path) -> tuple[int, int]:
    """Tier2: 只写四张核心表 + 月/年聚合（避开 risk/pollutant 等大表）。"""
    dest.mkdir(parents=True, exist_ok=True)
    tables = {
        "system": (result.system_daily, []),
        "component": (result.component_daily, ["component_id", "kind"]),
        "area": (result.area_daily, ["area_id"]),
        "data_center": (result.data_center_daily, ["data_center_id"]),
    }
    rows = 0
    size = 0

    def _dump(path: Path) -> None:
        nonlocal rows, size
        size += path.stat().st_size
        rows += max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)

    for name, (frame, identifiers) in tables.items():
        if frame.empty:
            continue
        daily = dest / f"{name}_daily.csv"
        frame.to_csv(daily, index=False)
        _dump(daily)
        monthly = _aggregate_frame(frame, "MS", identifiers)
        monthly_path = dest / f"{name}_monthly.csv"
        monthly.to_csv(monthly_path, index=False)
        _dump(monthly_path)
        annual = _aggregate_frame(frame, "YS", identifiers)
        annual_path = dest / f"{name}_annual.csv"
        annual.to_csv(annual_path, index=False)
        _dump(annual_path)
    return rows, size


def run_domain(domain: str, tiers: set[int]) -> list[dict]:
    base_project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
    manifest: list[dict] = []

    scenarios: list[tuple[str, str, dict, pd.DataFrame, int]] = []

    # ---- Tier1: 容量扫描点 -------------------------------------------------
    if 1 in tiers:
        for capacity in CAPACITIES:
            tag = f"cap_{int(capacity):04d}"
            scenarios.append((tag, "容量扫描", _set_capacity(base_project, capacity), timeseries, 1))

    # ---- Tier2: 冷却技术 / 分配政策 / 外部压力 / 能源富集 -------------------
    if 2 in tiers:
        for tech_file in sorted((R2 / domain).glob("tech_*.json")):
            tech = tech_file.stem.replace("tech_", "")
            proj = json.loads(tech_file.read_text(encoding="utf-8"))
            scenarios.append((f"tech_{tech}", "冷却技术", _set_capacity(proj, 750.0), timeseries, 2))
        for policy_file in sorted((R2 / domain).glob("policy_*.json")):
            policy = policy_file.stem.replace("policy_", "")
            proj = json.loads(policy_file.read_text(encoding="utf-8"))
            scenarios.append((f"policy_{policy}", "分配政策", _set_capacity(proj, 1000.0), timeseries, 2))
        for pressure in ("G1", "G2", "G3"):
            proj, drivers = _apply_g(base_project, timeseries, pressure)
            scenarios.append((f"press_{pressure.lower()}", "外部压力", _set_capacity(proj, 1000.0), drivers, 2))
        energy = R2 / domain / "energy_rich.json"
        if energy.exists():
            proj = json.loads(energy.read_text(encoding="utf-8"))
            scenarios.append(("energy_rich", "能源富集", _set_capacity(proj, 1000.0), timeseries, 2))

    print(f"[{domain}] {len(scenarios)} 个情景")
    for index, (tag, kind, project, drivers, tier) in enumerate(scenarios, start=1):
        started = time.time()
        dest = OUT / domain / tag
        result = FullAIUWMModel(project, drivers).run()
        rows, size = write_full(result, dest) if tier == 1 else write_slim(result, dest)
        kpis = summarize_ai_water_kpis(result, project)
        elapsed = time.time() - started

        dc = result.data_center_daily
        it_mwh = float(kpis.get("it_energy_mwh") or 0.0)
        manifest.append({
            "域": domain,
            "域标签": DOMAIN_LABEL[domain],
            "情景": tag,
            "情景类型": kind,
            "层级": f"Tier{tier}",
            "AI容量_MW": float(next(
                (c.get("installed_it_capacity_mw", 0.0)
                 for c in project["components"].values() if c.get("kind") == "data_center"),
                0.0,
            )),
            "总取水量_ML": float(kpis.get("total_withdrawal_ml") or 0.0),
            "耗水量_ML": float(kpis.get("consumption_ml") or 0.0),
            "WUE_L_per_kWh": (float(kpis.get("total_withdrawal_ml") or 0.0) * 1000.0 / it_mwh) if it_mwh else float("nan"),
            "PUE": float(dc["pue"].mean()) if not dc.empty else float("nan"),
            "城市未满足量_ML": float(result.system_daily["unmet_demand_ml"].sum()) if not result.system_daily.empty else float("nan"),
            "居民未满足量_ML": float(result.area_daily["unmet_domestic_ml"].sum()) if not result.area_daily.empty else float("nan"),
            "行数": rows,
            "大小_MB": round(size / 1024 / 1024, 3),
            "耗时_秒": round(elapsed, 2),
            "路径": str(dest.relative_to(OUT)).replace("\\", "/"),
        })
        print(f"  [{index:2d}/{len(scenarios)}] {tag:22s} rows={rows:7d} {size/1024/1024:6.2f}MB {elapsed:5.1f}s")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--tier", default="1,2")
    args = parser.parse_args()

    tiers = {int(t) for t in args.tier.split(",")}
    OUT.mkdir(parents=True, exist_ok=True)

    all_manifest: list[dict] = []
    total_rows = 0
    total_mb = 0.0
    for domain in args.domains.split(","):
        records = run_domain(domain.strip(), tiers)
        all_manifest.extend(records)
        for record in records:
            total_rows += record["行数"]
            total_mb += record["大小_MB"]

    frame = pd.DataFrame(all_manifest)

    # manifest 累积式合并：多次运行不同 --domains 时保留已有域的记录
    manifest_path = OUT / "manifest.csv"
    if manifest_path.exists():
        try:
            previous = pd.read_csv(manifest_path, encoding="utf-8-sig")
            if not previous.empty and "域" in previous.columns:
                kept = previous[~previous["域"].isin(set(frame["域"]))]
                frame = pd.concat([kept, frame], ignore_index=True)
        except Exception:
            pass

    frame = frame.sort_values(["域", "情景类型", "情景"]).reset_index(drop=True)
    frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    total_rows = int(frame["行数"].sum())
    total_mb = float(frame["大小_MB"].sum())
    print(f"\nmanifest 累计 {len(frame)} 个情景, {total_rows:,} 行, {total_mb:.1f} MB")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
