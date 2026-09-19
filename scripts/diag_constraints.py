"""Diagnostic: which constraint fails first at low AI capacity."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiuwm.full_engine import FullAIUWMModel
from aiuwm.ai_metrics import summarize_ai_water_kpis
from aiuwm.research import scan_ai_capacity

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"

DOMAIN = "ha"


def main() -> None:
    project = json.loads((R2 / DOMAIN / "project.json").read_text(encoding="utf-8"))
    constraints = json.loads(
        (R2 / "specs" / f"constraints_{DOMAIN}.json").read_text(encoding="utf-8")
    )["constraints"]
    ts = pd.read_csv(R2 / DOMAIN / "timeseries.csv")
    scan = scan_ai_capacity(project, ts, capacities_mw=[0, 100, 200, 400, 600, 800, 1000])
    cols = [c for c in scan.columns if c in constraints or c.endswith("_ml") or "ratio" in c
            or "util" in c or "reliab" in c or "unmet" in c]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(scan[["ai_capacity_mw"] + cols].to_string(index=False))
    print("\nconstraints:", json.dumps(constraints, ensure_ascii=False))
    for name, spec in constraints.items():
        if name in scan.columns:
            series = scan[name].astype(float)
            op, thr = spec["operator"], spec["value"]
            ok = series >= thr if op == ">=" else series <= thr
            first_fail = scan.loc[~ok, "ai_capacity_mw"].min()
            print(f"  {name:42s} {op} {thr:<8} first_fail_mw={first_fail}")


if __name__ == "__main__":
    main()
