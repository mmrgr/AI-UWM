"""Diagnostic: daily peaks needed to size abstraction capacity and reservoir."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from aiuwm.full_engine import FullAIUWMModel

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"


def main() -> None:
    for domain in ("ha", "co"):
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        ts = pd.read_csv(R2 / domain / "timeseries.csv")
        for mw in (0.0, 1000.0):
            trial = copy.deepcopy(project)
            if mw == 0.0:
                trial["components"].pop("AI_DC1", None)
                trial["supply_paths"][0]["demand_priority"] = [
                    i for i in trial["supply_paths"][0]["demand_priority"]
                    if not i.startswith("data_center::")
                ]
            else:
                trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = mw
            res = FullAIUWMModel(trial, ts).run()
            comp = res.component_daily
            absr = comp[comp["component_id"] == "RES1"]["outflow_ml"]
            wtw = comp[comp["component_id"] == "WTW1"]["inflow_ml"]
            wwtw = comp[comp["component_id"] == "WWTW1"]["inflow_ml"]
            store = comp[comp["component_id"] == "RES1"]["storage_ml"]
            print(f"{domain} {mw:>6g} MW | abstraction mean {absr.mean():7.2f} max {absr.max():7.2f}"
                  f" | wtw max {wtw.max():7.2f} | wwtw max {wwtw.max():7.2f}"
                  f" | storage min {store.min():8.1f} max {store.max():8.1f}"
                  f" | unmet {res.area_daily['unmet_ml'].sum():.3f}")


if __name__ == "__main__":
    main()
