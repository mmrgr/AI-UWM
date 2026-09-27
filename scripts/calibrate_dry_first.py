"""Calibrate the `dry_first` cooling archetype against literature peak factors.

Target (Han/Li/Wierman/Ren 2026, arXiv:2603.02705): a dry-cooling-dominant data
centre that only calls on evaporative assistance during hot periods shows a
daily peak-to-mean water-use factor of roughly 6.5-10x, versus ~2.2x for a
purely evaporative tower and ~1.2x for residential/commercial demand.

The repository's `dry_first` variant is implemented through the hybrid
wet-bulb threshold switch (`hybrid_wet_bulb_start_c` / `hybrid_full_wet_bulb_c`).
This script sweeps that switching band and reports, for both parameter-domain
endpoints, the resulting annual withdrawal and daily peak factor so the band can
be chosen to land inside the literature interval.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiuwm.full_engine import FullAIUWMModel

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"

# (start, full) candidates for the wet-bulb switching band, in degrees Celsius.
BANDS = [(22.0, 30.0), (21.0, 29.0), (20.0, 29.0), (20.0, 28.0), (19.0, 28.0), (18.0, 28.0), (18.0, 27.0)]

DRY_FIRST = {
    "technology": "hybrid",
    "technology_pue_adjustment": 0.10,
    "base_pue": 1.30,
    "temperature_coefficient": 0.0110,
    "cycles_of_concentration": 6.0,
}


def run_case(project: dict, timeseries: pd.DataFrame, start: float, full: float, capacity: float) -> dict:
    trial = copy.deepcopy(project)
    dc = None
    for component in trial["components"].values():
        if component.get("kind") == "data_center":
            dc = component
    dc["installed_it_capacity_mw"] = float(capacity)
    dc.pop("capacity_schedule", None)
    dc["base_pue"] = DRY_FIRST["base_pue"]
    dc["temperature_coefficient"] = DRY_FIRST["temperature_coefficient"]
    cooling = dc["cooling"]
    cooling["technology"] = DRY_FIRST["technology"]
    cooling["technology_pue_adjustment"] = {"hybrid": DRY_FIRST["technology_pue_adjustment"]}
    cooling["cycles_of_concentration"] = DRY_FIRST["cycles_of_concentration"]
    cooling["hybrid_wet_bulb_start_c"] = start
    cooling["hybrid_full_wet_bulb_c"] = full
    result = FullAIUWMModel(trial, timeseries).run()
    daily = result.data_center_daily.groupby("date", as_index=False).sum(numeric_only=True)
    series = daily["external_withdrawal_ml"].astype(float)
    mean = float(series.mean())
    years = max(len(timeseries) / 365.0, 1e-12)
    return {
        "start_c": start,
        "full_c": full,
        "band_width_c": full - start,
        "period_withdrawal_ml": float(series.sum()),
        "annual_withdrawal_ml": float(series.sum()) / years,
        "mean_wet_fraction": float(result.data_center_daily["wet_cooling_fraction"].mean()),
        "daily_peak_factor": float(series.max() / mean) if mean > 0 else float("nan"),
        "wet_days_per_year": int((result.data_center_daily["wet_cooling_fraction"] > 0.05).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--capacity", type=float, default=1000.0)
    args = parser.parse_args()

    rows = []
    for domain in args.domains.split(","):
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
        for start, full in BANDS:
            record = run_case(project, timeseries, start, full, args.capacity)
            record["domain"] = domain
            rows.append(record)
    frame = pd.DataFrame(rows)
    frame = frame[["domain", "start_c", "full_c", "band_width_c", "annual_withdrawal_ml",
                   "mean_wet_fraction", "wet_days_per_year", "daily_peak_factor"]]
    out = ROOT / "validation_artifacts" / "r2" / "dry_first_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.to_string(index=False))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
