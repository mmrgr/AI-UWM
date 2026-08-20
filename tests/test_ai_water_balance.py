from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from collections import defaultdict

from watermet2_repro.full_engine import FullWaterMet2Model, load_project


PROJECT = Path("examples/demo_full/project.json")


def ai_project(days: int = 5, capacity_mw: float = 500.0):
    project, timeseries = load_project(PROJECT)
    project = copy.deepcopy(project)
    timeseries = timeseries.iloc[:days].copy()
    project["simulation"]["end"] = str(timeseries["date"].iloc[-1].date())
    project["pipeline_events"] = []
    project["interventions"] = []
    project["components"]["AI_DC1"] = {
        "kind": "data_center",
        "local_area": "LA1",
        "installed_it_capacity_mw": capacity_mw,
        "load_factor": 0.75,
        "pue_mode": "dynamic",
        "base_pue": 1.2,
        "temperature_coefficient": 0.003,
        "min_pue": 1.05,
        "max_pue": 1.6,
        "cooling_storage_ml": 2.0,
        "initial_cooling_storage_ml": 2.0,
        "cooling": {
            "technology": "evaporative",
            "cycles_of_concentration": 5.0,
            "drift_fraction_of_makeup": 0.0002,
            "blowdown_return_fraction": 1.0,
            "internal_recovery_fraction": 0.0,
            "blowdown_quality_mg_l": {"TDS": 1600.0},
        },
        "water_sources": {
            "reclaimed": {
                "component_id": "CENTRAL_REUSE",
                "target_fraction": 0.70,
                "priority": 1,
            },
            "potable": {"target_fraction": 0.30, "priority": 2},
        },
        "water_fallback": True,
        "priority_class": "ai_data_center",
    }
    return project, timeseries


def test_zero_ai_capacity_is_exact_baseline() -> None:
    baseline, timeseries = ai_project(capacity_mw=0.0)
    without_ai = copy.deepcopy(baseline)
    without_ai["components"].pop("AI_DC1")
    baseline_result = FullWaterMet2Model(without_ai, timeseries).run()
    ai_result = FullWaterMet2Model(baseline, timeseries).run()
    np.testing.assert_allclose(
        ai_result.system_daily["water_demand_ml"],
        baseline_result.system_daily["water_demand_ml"],
        atol=0.0,
    )
    assert (ai_result.data_center_daily["external_withdrawal_ml"] == 0).all()


def test_ai_is_supplied_by_real_reuse_and_potable_and_returns_to_wwtw() -> None:
    project, timeseries = ai_project()
    result = FullWaterMet2Model(project, timeseries).run()
    dc = result.data_center_daily
    assert len(dc) == len(timeseries)
    assert dc["it_energy_mwh"].sum() > 0
    assert dc["external_withdrawal_ml"].sum() > 0
    assert dc["reclaimed_water_ml"].sum() > 0
    assert (
        dc["reclaimed_water_ml"]
        <= dc["external_withdrawal_ml"] * dc["target_reclaimed_fraction"] + 1e-10
    ).all()
    np.testing.assert_allclose(dc["water_balance_residual_ml"], 0.0, atol=1e-9)
    sewer = result.component_daily[result.component_daily.component_id == "SEWER1"]
    without_ai = copy.deepcopy(project)
    without_ai["components"].pop("AI_DC1")
    baseline = FullWaterMet2Model(without_ai, timeseries).run()
    baseline_sewer = baseline.component_daily[
        baseline.component_daily.component_id == "SEWER1"
    ]
    assert sewer["inflow_ml"].sum() > baseline_sewer["inflow_ml"].sum()
    tds = result.pollutant_daily[
        (result.pollutant_daily.component_id == "AI_DC1")
        & (result.pollutant_daily.pollutant == "TDS")
    ]
    assert tds["mass_kg"].sum() > 0


def test_reclaimed_supply_is_limited_by_reuse_storage() -> None:
    project, timeseries = ai_project(days=1)
    project["components"]["CENTRAL_REUSE"]["initial_ml"] = 0.25
    project["components"]["CENTRAL_REUSE"]["treatment_capacity_ml_day"] = 0.25
    result = FullWaterMet2Model(project, timeseries).run()
    assert result.data_center_daily.iloc[0]["reclaimed_water_ml"] <= 0.25 + 1e-12


def test_no_fallback_preserves_unmet_water_and_other_source_is_accounted() -> None:
    project, timeseries = ai_project(days=1)
    project["components"]["CENTRAL_REUSE"]["initial_ml"] = 0.0
    project["components"]["AI_DC1"]["water_fallback"] = False
    limited = FullWaterMet2Model(project, timeseries).run().data_center_daily.iloc[0]
    assert limited["unmet_cooling_water_ml"] > 0

    project, timeseries = ai_project(days=1)
    sources = project["components"]["AI_DC1"]["water_sources"]
    sources["reclaimed"]["target_fraction"] = 0.4
    sources["potable"]["target_fraction"] = 0.3
    sources["other"] = {"target_fraction": 0.3, "available_ml_day": 100}
    supplied = FullWaterMet2Model(project, timeseries).run().data_center_daily.iloc[0]
    assert supplied["other_water_ml"] > 0
    assert abs(supplied["water_balance_residual_ml"]) < 1e-9


def test_shortage_allocation_supports_resident_proportional_and_ai_first() -> None:
    def state():
        return {
            "demand": {"domestic": 10.0, "data_center::AI": 10.0},
            "remaining": {"domestic": 10.0, "data_center::AI": 10.0},
            "potable_delivered": defaultdict(float),
            "minimum_service_fractions": {},
        }

    resident = state()
    FullWaterMet2Model._allocate_potable_to_demands(resident, 10, None, "resident_first")
    assert resident["potable_delivered"]["domestic"] == 10
    proportional = state()
    FullWaterMet2Model._allocate_potable_to_demands(proportional, 10, None, "proportional")
    assert proportional["potable_delivered"]["domestic"] == 5
    ai_first = state()
    FullWaterMet2Model._allocate_potable_to_demands(ai_first, 10, None, "ai_first")
    assert ai_first["potable_delivered"]["data_center::AI"] == 10
