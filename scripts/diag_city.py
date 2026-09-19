"""Diagnostic: inspect city-side water-balance columns and reuse flows."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiuwm.full_engine import FullAIUWMModel

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"


def main() -> None:
    for domain in ("ha", "co"):
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        ts = pd.read_csv(R2 / domain / "timeseries.csv")
        base = copy.deepcopy(project)
        base["components"].pop("AI_DC1", None)
        base["supply_paths"][0]["demand_priority"] = [
            i for i in base["supply_paths"][0]["demand_priority"]
            if not i.startswith("data_center::")
        ]
        res = FullAIUWMModel(base, ts).run()

        print(f"\n{'=' * 78}\n{domain}: area_daily columns")
        print(list(res.area_daily.columns))
        cols = [c for c in res.area_daily.columns
                if c.startswith("demand") or c.startswith("delivered") or "reuse" in c]
        print(res.area_daily[cols].mean().round(3).to_string())

        print(f"\n{domain}: component_daily means (ML/d)")
        comp = res.component_daily
        g = comp.groupby("component_id").mean(numeric_only=True)
        print(g.round(3).to_string())

        print(f"\n{domain}: CENTRAL_REUSE daily (first 400 d, every 60)")
        cr = comp[comp["component_id"] == "CENTRAL_REUSE"].reset_index(drop=True)
        print(cr[["inflow_ml", "outflow_ml", "delivered_ml", "storage_ml"]].iloc[::60].round(2).to_string())
        print("storage max/min:", cr["storage_ml"].max(), cr["storage_ml"].min())
        print("\nWWTW1 columns:", list(comp.columns))
        ww = comp[comp["component_id"] == "WWTW1"].reset_index(drop=True)
        print(ww.mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
