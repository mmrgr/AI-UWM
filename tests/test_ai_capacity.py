from __future__ import annotations

import copy
from pathlib import Path
import pandas as pd

from watermet2_repro.ai_capacity import find_ai_carrying_capacity, scan_ai_capacity
from watermet2_repro.full_engine import load_project


def project_with_ai():
    project, timeseries = load_project(Path("examples/demo_full/project.json"))
    project = copy.deepcopy(project)
    timeseries = timeseries.iloc[:4].copy()
    project["simulation"]["end"] = str(timeseries["date"].iloc[-1].date())
    project["interventions"] = []
    project["pipeline_events"] = []
    project["components"]["AI_DC1"] = {
        "kind": "data_center",
        "local_area": "LA1",
        "installed_it_capacity_mw": 0,
        "load_factor": 0.8,
        "base_pue": 1.2,
        "cooling": {"technology": "evaporative", "cycles_of_concentration": 5},
        "water_sources": {"reclaimed": {"component_id": "CENTRAL_REUSE", "target_fraction": 0.5}, "potable": {"target_fraction": 0.5}},
    }
    return project, timeseries


def test_capacity_scan_is_monotonic_for_energy_and_water() -> None:
    project, timeseries = project_with_ai()
    scan = scan_ai_capacity(project, timeseries, capacities_mw=[0, 100, 300])
    assert scan["ai_capacity_mw"].tolist() == [0, 100, 300]
    assert scan["it_energy_mwh"].is_monotonic_increasing
    assert scan["total_withdrawal_ml"].is_monotonic_increasing


def test_threshold_finder_reports_known_boundary() -> None:
    project, timeseries = project_with_ai()
    scan = scan_ai_capacity(project, timeseries, capacities_mw=[0, 100, 200, 300])
    limit = float(scan.loc[scan.ai_capacity_mw == 200, "total_withdrawal_ml"].iloc[0])
    threshold = find_ai_carrying_capacity(
        scan, {"total_withdrawal_ml": {"operator": "<=", "value": limit}}
    )
    assert threshold["maximum_safe_ai_capacity_mw"] == 200
    assert threshold["first_failed_capacity_mw"] == 300
    assert threshold["limiting_constraint"] == "total_withdrawal_ml"


def test_nonmonotone_scan_keeps_first_failure_and_all_feasible_intervals() -> None:
    scan = pd.DataFrame({
        "ai_capacity_mw": [0, 100, 200, 300, 400],
        "stress": [0, 0, 2, 0, 3],
    })
    threshold = find_ai_carrying_capacity(
        scan, {"stress": {"operator": "<=", "value": 0}}
    )
    assert threshold["first_failed_capacity_mw"] == 200
    assert threshold["threshold_interval_mw"] == [100, 200]
    assert threshold["maximum_safe_ai_capacity_mw"] == 300
    assert threshold["feasible_intervals_mw"] == [[0, 100], [300, 300]]
    assert threshold["reentrant_feasibility"] is True
