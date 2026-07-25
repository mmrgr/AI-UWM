from __future__ import annotations

from pathlib import Path
import copy

import numpy as np
import pandas as pd

from watermet2_repro.analysis import summarize_kpis
from watermet2_repro.full_engine import FullWaterMet2Model, load_project


def main() -> None:
    output = Path("examples/demo_full")
    output.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2011-01-01", "2012-12-31", freq="D")
    rng = np.random.default_rng(3302)
    day = dates.dayofyear.to_numpy()
    temperature = 6.5 + 10.5 * np.sin(2 * np.pi * (day - 172) / 365.25) + rng.normal(0, 3, len(dates))
    wet = rng.random(len(dates)) < 0.45
    rainfall = np.where(wet, rng.gamma(1.25, 4.2, len(dates)), 0.0)
    wr1 = (287_000 / 365.25) * (1 + 0.35 * np.sin(2 * np.pi * (day - 105) / 365.25))
    wr2 = (12_000 / 365.25) * (1 + 0.35 * np.sin(2 * np.pi * (day - 105) / 365.25))
    frame = pd.DataFrame(
        {
            "date": dates,
            "temperature_c": temperature,
            "rainfall_mm": rainfall,
            "wr1_inflow_ml": np.maximum(0, wr1),
            "wr2_inflow_ml": np.maximum(0, wr2),
        }
    )
    frame.to_csv(output / "timeseries.csv", index=False)
    result = FullWaterMet2Model.from_files(output / "project.json").run()
    sewer = result.component_daily[result.component_daily["component_id"] == "SEWER1"]
    observed = sewer[["date", "outflow_ml"]].rename(
        columns={"outflow_ml": "observed_sewer_outflow_ml"}
    )
    observed["observed_sewer_outflow_ml"] *= 1 + rng.normal(0, 0.02, len(observed))
    observed["observed_sewer_outflow_ml"] = observed["observed_sewer_outflow_ml"].clip(lower=0)
    observed.to_csv(output / "observed.csv", index=False)

    project, timeseries = load_project(output / "project.json")
    scenarios = []
    for name, changes in (
        ("BAU", {}),
        ("Leakage reduction", {"components.DM1.leakage_fraction": 0.15}),
        (
            "Enhanced reuse",
            {
                "components.RWH_LOCAL.collection_fraction": 0.8,
                "components.GWR_SUB.collection_fraction": 0.8,
            },
        ),
    ):
        scenario = copy.deepcopy(project)
        for path, value in changes.items():
            target = scenario
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        record = {"alternative": name}
        record.update(summarize_kpis(FullWaterMet2Model(scenario, timeseries).run()))
        scenarios.append(record)
    pd.DataFrame(scenarios).to_csv(output / "alternatives.csv", index=False)


if __name__ == "__main__":
    main()
