"""Run a longer synthetic hydrological stress diagnostic without changing R2 inputs.

This keeps the published two-year R2 files untouched, generates a deterministic
10-year driver from the same endpoint rules, and reports terminal-storage and
resident-reliability gates separately from the short-term facility boundary.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from aiuwm.full_engine import FullAIUWMModel  # noqa: E402
from build_r2_domains import DOMAINS, base_project, build_drivers  # noqa: E402

OUT = ROOT / "validation_artifacts" / "r2" / "reservoir_multiyear_10y.json"


def audit_domain(domain: str, years: int, max_mw: float, step_mw: float, seed: int) -> dict:
    domain_spec = DOMAINS[domain]
    offset = 101 * list(DOMAINS).index(domain)
    timeseries = build_drivers(domain_spec, "2029-01-01", years, seed + offset)
    project = base_project(domain, domain_spec, years=years)
    # build_drivers uses a fixed 365-day synthetic year; align the project end
    # date with the generated driver rather than claiming leap-day coverage.
    project["simulation"]["end"] = str(pd.Timestamp(timeseries["date"].iloc[-1]).date())
    data_center_id = next(
        key for key, value in project["components"].items() if value.get("kind") == "data_center"
    )
    capacities = [float(v) for v in range(0, int(max_mw) + 1, int(step_mw))]
    initial = float(project["components"]["RES1"]["initial_ml"])
    rows: list[dict] = []
    for capacity in capacities:
        trial = copy.deepcopy(project)
        trial["components"][data_center_id]["installed_it_capacity_mw"] = capacity
        trial["components"][data_center_id].pop("capacity_schedule", None)
        result = FullAIUWMModel(trial, timeseries).run()
        storage = result.component_daily.loc[
            result.component_daily["component_id"] == "RES1", "storage_ml"
        ].astype(float)
        system = result.system_daily
        system_demand = float(system["water_demand_ml"].sum())
        system_unmet = float(system["unmet_demand_ml"].sum())
        # The system row includes residents, industry, irrigation and
        # municipal demand.  Use the explicit domestic columns for the
        # resident-service gate; retain the all-demand ratio as a separate
        # diagnostic so the two estimands cannot be confused.
        area = result.area_daily
        resident_demand = float(area["demand_domestic_ml"].sum()) if "demand_domestic_ml" in area else 0.0
        resident_unmet = float(area["unmet_domestic_ml"].sum()) if "unmet_domestic_ml" in area else 0.0
        resident_reliability = 1.0 - resident_unmet / resident_demand if resident_demand else 1.0
        system_reliability = 1.0 - system_unmet / system_demand if system_demand else 1.0
        terminal = float(storage.iloc[-1])
        rows.append({
            "ai_capacity_mw": capacity,
            "initial_storage_ml": initial,
            "terminal_storage_ml": terminal,
            "terminal_storage_fraction": terminal / initial,
            "minimum_storage_ml": float(storage.min()),
            "minimum_storage_fraction": float(storage.min()) / initial,
            "resident_demand_ml": resident_demand,
            "resident_unmet_demand_ml": resident_unmet,
            "resident_reliability_fraction": resident_reliability,
            "system_demand_ml": system_demand,
            "system_unmet_demand_ml": system_unmet,
            "system_reliability_fraction": system_reliability,
            "terminal_ge_100pct": terminal >= initial,
            "terminal_ge_90pct": terminal >= 0.90 * initial,
            "reliability_ge_99pct": resident_reliability >= 0.99,
        })
    frame = pd.DataFrame(rows)

    # A capacity scan is interpreted as a first-failure threshold only when
    # the gate does not re-enter at higher capacities.
    gate = frame["terminal_ge_90pct"] & frame["reliability_ge_99pct"]
    transitions = (gate.astype(int).diff().fillna(0) != 0).sum()
    if transitions > 1:
        raise RuntimeError(f"{domain}: joint gate re-enters across the capacity grid")

    def last_pass(column: str) -> float | None:
        values = frame.loc[frame[column], "ai_capacity_mw"]
        return float(values.max()) if not values.empty else None

    gate_values = frame.loc[gate, "ai_capacity_mw"]
    return {
        "domain": domain,
        "years": years,
        "seed": seed + offset,
        "driver_basis": "365-day synthetic years; fixed 365*years daily records, not Gregorian calendar years",
        "hydrology_basis": "single deterministic synthetic sequence generated from build_r2_domains endpoint rules",
        "initial_storage_fraction_of_capacity": initial / float(project["components"]["RES1"]["capacity_ml"]),
        "criterion_100pct": "terminal storage >= initial storage",
        "criterion_90pct_and_reliability": "terminal storage >= 90% initial and resident-service reliability >= 99%",
        "capacity_step_mw": step_mw,
        "last_terminal_ge_100pct_mw": last_pass("terminal_ge_100pct"),
        "last_terminal_ge_90pct_mw": last_pass("terminal_ge_90pct"),
        "last_joint_gate_mw": float(gate_values.max()) if not gate_values.empty else None,
        "first_joint_gate_failure_mw": float(frame.loc[~gate, "ai_capacity_mw"].min()) if (~gate).any() else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--max-mw", type=float, default=2000.0)
    parser.add_argument("--step-mw", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    reports = {
        domain.strip(): audit_domain(domain.strip(), args.years, args.max_mw, args.step_mw, args.seed)
        for domain in args.domains.split(",")
    }
    OUT.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports.values():
        print(
            report["domain"],
            "joint_gate", report["last_joint_gate_mw"],
            "terminal100", report["last_terminal_ge_100pct_mw"],
        )


if __name__ == "__main__":
    main()
