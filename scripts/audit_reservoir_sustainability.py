"""Audit whether the two-year reservoir trajectory is non-depleting.

This is a diagnostic, not a replacement for a multi-year hydrological run.  It
uses the same engine and 50 MW grid as the R2 deterministic scan and compares
terminal storage with the configured initial storage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aiuwm.full_engine import FullAIUWMModel


def audit_domain(domain: str, max_mw: float, step_mw: float) -> dict:
    folder = ROOT / "examples" / "cawcc_r2" / domain
    project = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(folder / "timeseries.csv")
    data_center_id = next(key for key, value in project["components"].items() if value.get("kind") == "data_center")
    capacities = [float(v) for v in range(0, int(max_mw) + 1, int(step_mw))]
    initial = float(project["components"]["RES1"]["initial_ml"])
    rows: list[dict] = []
    for capacity in capacities:
        trial = json.loads(json.dumps(project))
        trial["components"][data_center_id]["installed_it_capacity_mw"] = capacity
        trial["components"][data_center_id].pop("capacity_schedule", None)
        result = FullAIUWMModel(trial, timeseries).run()
        storage = result.component_daily.loc[
            result.component_daily["component_id"] == "RES1", ["date", "storage_ml"]
        ].copy()
        storage["date"] = pd.to_datetime(storage["date"])
        first_year = storage[storage["date"].dt.year == storage["date"].dt.year.min()]
        last_year = storage[storage["date"].dt.year == storage["date"].dt.year.max()]
        terminal = float(storage.iloc[-1]["storage_ml"])
        rows.append({
            "ai_capacity_mw": capacity,
            "initial_storage_ml": initial,
            "terminal_storage_ml": terminal,
            "terminal_minus_initial_ml": terminal - initial,
            "terminal_storage_fraction": terminal / initial,
            "minimum_storage_ml": float(storage["storage_ml"].min()),
            "first_year_net_change_ml": float(first_year.iloc[-1]["storage_ml"] - initial),
            "second_year_net_change_ml": float(last_year.iloc[-1]["storage_ml"] - first_year.iloc[-1]["storage_ml"]),
            "terminal_non_depleting": terminal >= initial,
        })
    frame = pd.DataFrame(rows)
    non_depleting = frame.loc[frame["terminal_non_depleting"], "ai_capacity_mw"]
    first_decline = frame.loc[~frame["terminal_non_depleting"], "ai_capacity_mw"]
    return {
        "domain": domain,
        "criterion": "terminal_storage_ml >= configured initial_storage_ml after 730 days",
        "initial_storage_ml": initial,
        "capacity_step_mw": step_mw,
        "max_terminal_non_depleting_capacity_mw": float(non_depleting.max()) if not non_depleting.empty else None,
        "first_terminal_decline_capacity_mw": float(first_decline.min()) if not first_decline.empty else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--max-mw", type=float, default=2000.0)
    parser.add_argument("--step-mw", type=float, default=50.0)
    args = parser.parse_args()
    reports = {domain: audit_domain(domain, args.max_mw, args.step_mw) for domain in args.domains.split(",")}
    out = ROOT / "validation_artifacts" / "r2" / "reservoir_sustainability_corrected.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports.values():
        print(report["domain"], report["max_terminal_non_depleting_capacity_mw"], report["first_terminal_decline_capacity_mw"])


if __name__ == "__main__":
    main()
