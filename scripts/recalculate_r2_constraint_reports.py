"""Rebuild R2 constraint labels from existing KPI scans without rerunning the model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from aiuwm.ai_capacity import find_ai_carrying_capacity, summarize_constraint_boundaries  # noqa: E402
from run_r2 import daily_constraints, annual_constraints, hourly_cawcc  # noqa: E402
from aiuwm.cawcc import build_intraday_profile  # noqa: E402

R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"


def main() -> None:
    for domain in ("ha", "co"):
        folder = OUT / domain
        scan = pd.read_csv(folder / "capacity_scan.csv")
        raw = json.loads((R2 / "specs" / f"constraints_{domain}.json").read_text(encoding="utf-8"))
        daily = {"constraints": daily_constraints(raw)}
        annual = {"constraints": annual_constraints(raw)}
        positive = scan[scan["ai_capacity_mw"] >= 50].reset_index(drop=True)
        threshold = find_ai_carrying_capacity(positive, daily)
        threshold["capacity_scan"].to_csv(folder / "capacity_scan_corrected.csv", index=False)
        summarize_constraint_boundaries(positive, daily).to_csv(
            folder / "constraint_boundaries_corrected.csv", index=False
        )
        annual_threshold = find_ai_carrying_capacity(positive, annual)
        physical_daily = {"constraints": {
            k: v for k, v in daily["constraints"].items()
            if k != "reclaimed_water_substitution_ratio"
        }}
        physical_annual = {"constraints": {
            k: v for k, v in annual["constraints"].items()
            if k != "reclaimed_water_substitution_ratio"
        }}
        policy = {"constraints": {
            "reclaimed_water_substitution_ratio": daily["constraints"]["reclaimed_water_substitution_ratio"]
        }}
        profile = build_intraday_profile(
            training_fraction=0.65, inference_fraction=0.35,
            inference_peak_factor=1.35, peak_hours=range(10, 22),
        )
        physical_hourly = hourly_cawcc(scan, physical_daily, profile)
        summary = {
            "domain": domain,
            "grid_mw": 50,
            "daily": {k: v for k, v in threshold.items() if k != "capacity_scan"},
            "annual": {k: v for k, v in annual_threshold.items() if k != "capacity_scan"},
            "physical_daily": {k: v for k, v in find_ai_carrying_capacity(positive, physical_daily).items() if k != "capacity_scan"},
            "physical_annual": {k: v for k, v in find_ai_carrying_capacity(positive, physical_annual).items() if k != "capacity_scan"},
            "physical_hourly": {k: v for k, v in physical_hourly.items() if k != "capacity_scan"},
            "reclaimed_target": {k: v for k, v in find_ai_carrying_capacity(positive, policy).items() if k != "capacity_scan"},
        }
        (folder / "capacity_threshold_corrected.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(domain, summary["daily"]["maximum_safe_ai_capacity_mw"], summary["daily"]["limiting_constraint"])


if __name__ == "__main__":
    main()
