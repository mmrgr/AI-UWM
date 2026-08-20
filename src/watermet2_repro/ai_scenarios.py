from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
import numpy as np

from .ai_metrics import summarize_ai_water_kpis
from .full_engine import FullWaterMet2Model


DEFAULT_CAPACITIES_MW = (100, 300, 500, 800, 1000, 1500, 2000)
DEFAULT_COOLING = ("evaporative", "efficient_evaporative", "hybrid", "dry", "liquid_to_air", "liquid_to_water")
DEFAULT_WATER_SOURCES = ("potable", "70_30", "50_50", "20_80", "reclaimed_first")
DEFAULT_HYDROCLIMATE = ("normal", "hot", "drought", "hot+drought")
DEFAULT_INFRASTRUCTURE = ("current", "reuse_expansion", "wtw_expansion", "leakage_reduction", "combined_upgrade")


def generate_ai_scenario_matrix(
    capacities_mw: Iterable[float] = DEFAULT_CAPACITIES_MW,
    cooling: Iterable[str] = DEFAULT_COOLING,
    water_sources: Iterable[str] = DEFAULT_WATER_SOURCES,
    hydroclimate: Iterable[str] = DEFAULT_HYDROCLIMATE,
    infrastructure: Iterable[str] = DEFAULT_INFRASTRUCTURE,
    *,
    include: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    scenarios = []
    for values in itertools.product(capacities_mw, cooling, water_sources, hydroclimate, infrastructure):
        scenario = dict(zip(("ai_capacity_mw", "cooling", "water_source", "hydroclimate", "infrastructure"), values))
        scenario["scenario_id"] = "__".join(map(str, values))
        if include is None or include(scenario):
            scenarios.append(scenario)
    return scenarios


def apply_ai_scenario(
    project: dict[str, Any], timeseries: pd.DataFrame, scenario: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    trial = copy.deepcopy(project)
    drivers = timeseries.copy()
    data_centers = [value for value in trial["components"].values() if value.get("kind") == "data_center"]
    if not data_centers:
        raise ValueError("Project contains no data_center component")
    for component in data_centers:
        component["installed_it_capacity_mw"] = float(scenario["ai_capacity_mw"])
        component.pop("capacity_schedule", None)
        component.setdefault("cooling", {})["technology"] = scenario["cooling"]
        reclaimed, potable = {
            "potable": (0.0, 1.0), "70_30": (0.7, 0.3), "50_50": (0.5, 0.5),
            "20_80": (0.2, 0.8), "reclaimed_first": (1.0, 0.0),
        }[scenario["water_source"]]
        sources = component.setdefault("water_sources", {})
        sources.setdefault("reclaimed", {})["target_fraction"] = reclaimed
        sources.setdefault("potable", {})["target_fraction"] = potable
        component["water_fallback"] = scenario["water_source"] == "reclaimed_first"

    hydro = scenario["hydroclimate"]
    if hydro in {"hot", "hot+drought"} and "temperature_c" in drivers:
        summer = pd.to_datetime(drivers["date"]).dt.month.isin((6, 7, 8))
        drivers.loc[summer, "temperature_c"] += float(scenario.get("summer_temperature_delta_c", 4.0))
    if hydro in {"drought", "hot+drought"}:
        factor = float(scenario.get("drought_inflow_factor", 0.65))
        for column in drivers.columns:
            if "inflow" in column.lower():
                drivers[column] *= factor

    infrastructure = scenario["infrastructure"]
    for component in trial["components"].values():
        kind = component.get("kind")
        if infrastructure in {"reuse_expansion", "combined_upgrade"} and kind == "reuse":
            for key in ("capacity_ml", "treatment_capacity_ml_day"):
                if key in component:
                    component[key] *= 1.5
        if infrastructure in {"wtw_expansion", "combined_upgrade"} and kind == "wtw":
            for key in ("capacity_ml", "daily_capacity_ml"):
                if key in component:
                    component[key] *= 1.25
        if infrastructure in {"leakage_reduction", "combined_upgrade"} and kind == "distribution_main":
            component["leakage_fraction"] = float(component.get("leakage_fraction", 0.0)) * 0.7
    return trial, drivers


def run_ai_scenario_matrix(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    scenarios: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        trial, drivers = apply_ai_scenario(project, timeseries, scenario)
        result = FullWaterMet2Model(trial, drivers).run()
        rows.append({**scenario, **summarize_ai_water_kpis(result, trial)})
    return pd.DataFrame(rows)


def compound_hot_drought_scenario(ai_capacity_mw: float, cooling: str = "evaporative") -> dict[str, Any]:
    return {
        "scenario_id": f"compound_hot_drought_{ai_capacity_mw:g}mw",
        "ai_capacity_mw": ai_capacity_mw,
        "cooling": cooling,
        "water_source": "70_30",
        "hydroclimate": "hot+drought",
        "infrastructure": "current",
    }


def pareto_ai_strategies(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    scenarios: Iterable[dict[str, Any]],
    objectives: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Evaluate discrete AI designs and flag non-dominated capacity/water/energy strategies."""
    objectives = objectives or {
        "ai_capacity_mw": "max",
        "freshwater_withdrawal_ml": "min",
        "unmet_cooling_water_ml": "min",
        "system_energy_kwh": "min",
    }
    frame = run_ai_scenario_matrix(project, timeseries, scenarios)
    missing = set(objectives) - set(frame)
    if missing:
        raise KeyError(f"Unknown Pareto objectives: {sorted(missing)}")
    signs = np.array([1.0 if direction == "min" else -1.0 for direction in objectives.values()])
    values = frame[list(objectives)].to_numpy(dtype=float) * signs
    pareto = np.ones(len(frame), dtype=bool)
    for index, candidate in enumerate(values):
        if (np.all(values <= candidate, axis=1) & np.any(values < candidate, axis=1)).any():
            pareto[index] = False
    frame["is_pareto"] = pareto
    return frame.sort_values(["is_pareto", "ai_capacity_mw"], ascending=[False, False]).reset_index(drop=True)
