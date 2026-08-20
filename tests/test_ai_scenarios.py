from __future__ import annotations

import copy
from pathlib import Path

from watermet2_repro.ai_scenarios import (
    apply_ai_scenario,
    compound_hot_drought_scenario,
    generate_ai_scenario_matrix,
    pareto_ai_strategies,
    run_ai_scenario_matrix,
)
from watermet2_repro.full_engine import load_project


def fixture():
    project, timeseries = load_project(Path("examples/demo_full/project.json"))
    project = copy.deepcopy(project)
    timeseries = timeseries.iloc[:3].copy()
    project["simulation"]["end"] = str(timeseries.date.iloc[-1].date())
    project["interventions"] = []
    project["pipeline_events"] = []
    project["components"]["AI_DC1"] = {
        "kind": "data_center", "local_area": "LA1", "installed_it_capacity_mw": 100,
        "load_factor": 0.8, "base_pue": 1.2,
        "cooling": {"technology": "evaporative", "cycles_of_concentration": 5},
        "water_sources": {"reclaimed": {"component_id": "CENTRAL_REUSE", "target_fraction": 0.7}, "potable": {"target_fraction": 0.3}},
    }
    return project, timeseries


def test_matrix_generation_and_filtering() -> None:
    matrix = generate_ai_scenario_matrix(
        capacities_mw=[100, 300], cooling=["evaporative", "dry"],
        water_sources=["70_30"], hydroclimate=["normal", "hot+drought"],
        infrastructure=["current"], include=lambda row: not (row["cooling"] == "dry" and row["water_source"] == "70_30"),
    )
    assert len(matrix) == 4


def test_compound_hot_drought_reduces_inflow_and_increases_heat() -> None:
    project, timeseries = fixture()
    scenario = compound_hot_drought_scenario(500)
    _trial, changed = apply_ai_scenario(project, timeseries, scenario)
    assert changed.filter(like="inflow").sum().sum() < timeseries.filter(like="inflow").sum().sum()


def test_scenario_runner_compares_cooling_technologies() -> None:
    project, timeseries = fixture()
    scenarios = generate_ai_scenario_matrix(
        capacities_mw=[300], cooling=["evaporative", "dry"], water_sources=["70_30"],
        hydroclimate=["normal"], infrastructure=["current"],
    )
    result = run_ai_scenario_matrix(project, timeseries, scenarios)
    wet = result[result.cooling == "evaporative"].iloc[0]
    dry = result[result.cooling == "dry"].iloc[0]
    assert wet.total_withdrawal_ml > dry.total_withdrawal_ml
    assert dry.facility_energy_mwh > wet.facility_energy_mwh
    pareto = pareto_ai_strategies(project, timeseries, scenarios)
    assert pareto.is_pareto.any()
