from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from watermet2_repro.analysis import (
    compromise_programming_rank,
    grid_calibrate,
    monte_carlo,
)
from watermet2_repro.full_engine import FullWaterMet2Model, load_project
from watermet2_repro.validation import ProjectValidationError, validate_project


PROJECT = Path("examples/demo_full/project.json")


@pytest.fixture(scope="module")
def demo() -> tuple[dict, pd.DataFrame, object]:
    project, timeseries = load_project(PROJECT)
    result = FullWaterMet2Model(project, timeseries).run()
    return project, timeseries, result


def test_complete_demo_exercises_all_component_families(demo: tuple[dict, pd.DataFrame, object]) -> None:
    project, _timeseries, result = demo
    kinds = {component["kind"] for component in project["components"].values()}
    assert {
        "water_resource",
        "supply_conduit",
        "wtw",
        "trunk_main",
        "service_reservoir",
        "distribution_main",
        "reuse",
        "sewer",
        "wwtw",
        "receiving_water",
    } <= kinds
    assert not result.recovery_daily.empty
    assert not result.pollutant_daily.empty
    assert len(result.material_events) == 1


def test_demand_and_resource_mass_balances_close(demo: tuple[dict, pd.DataFrame, object]) -> None:
    project, timeseries, result = demo
    np.testing.assert_allclose(
        result.area_daily["water_demand_ml"],
        result.area_daily["delivered_total_ml"] + result.area_daily["unmet_ml"],
        atol=1e-8,
    )
    for resource_id in ("WR1", "WR2"):
        component = project["components"][resource_id]
        rows = result.component_daily[result.component_daily["component_id"] == resource_id]
        expected_final = (
            component["initial_ml"]
            + timeseries[component["inflow_column"]].sum()
            - rows["outflow_ml"].sum()
            - rows["loss_ml"].sum()
            - rows["overflow_ml"].sum()
        )
        assert np.isclose(expected_final, rows["storage_ml"].iloc[-1], atol=1e-6)


def test_reuse_pollutants_recovery_and_impacts_are_reported(demo: tuple[dict, pd.DataFrame, object]) -> None:
    _project, _timeseries, result = demo
    assert result.area_daily["reuse_delivered_ml"].sum() > 0
    assert set(result.pollutant_daily["pollutant"]) >= {"BOD", "TSS", "TN", "TP"}
    assert {"biogas", "ammonium_nitrate", "generated_electricity"} <= set(result.recovery_daily["product"])
    assert result.system_daily["electricity_kwh"].sum() > 0
    assert result.system_daily["energy_generated_kwh"].sum() > 0
    assert result.system_daily["ghg_avoided_kg_co2e"].sum() > 0
    assert result.system_daily["ghg_caused_kg_co2e"].sum() > 0
    assert result.system_daily["eutrophication_caused_kg_po4e"].sum() > 0
    assert result.system_daily["capital_cost_eur"].sum() > 0
    for component_id in ("WTW1", "WTW2"):
        rows = result.component_daily[result.component_daily["component_id"] == component_id]
        np.testing.assert_allclose(rows["treated_ml"], rows["inflow_ml"], atol=1e-10)
    reuse = result.component_daily[result.component_daily["kind"] == "reuse"]
    np.testing.assert_allclose(reuse["treated_ml"], reuse["outflow_ml"], atol=1e-10)


def test_infiltration_and_exfiltration_are_in_component_water_balance(
    demo: tuple[dict, pd.DataFrame, object],
) -> None:
    project, timeseries, _result = demo
    short_timeseries = timeseries.iloc[:60].copy()
    trial = copy.deepcopy(project)
    trial["simulation"]["end"] = str(short_timeseries["date"].iloc[-1].date())
    trial["components"]["SEWER1"]["infiltration_fraction"] = 0.03
    trial["components"]["SEWER1"]["exfiltration_fraction"] = 0.01
    result = FullWaterMet2Model(trial, short_timeseries).run()
    sewer = result.component_daily[result.component_daily["component_id"] == "SEWER1"]
    residual = (
        sewer["inflow_ml"].sum()
        - sewer["outflow_ml"].sum()
        - sewer["loss_ml"].sum()
        - sewer["overflow_ml"].sum()
        - sewer["storage_ml"].iloc[-1]
    )
    assert abs(residual) < 1e-9
    assert sewer["infiltration_ml"].sum() > 0
    assert sewer["exfiltration_ml"].sum() > 0


def test_invalid_path_allocation_is_rejected(demo: tuple[dict, pd.DataFrame, object]) -> None:
    project, timeseries, _result = demo
    invalid = copy.deepcopy(project)
    invalid["supply_paths"][0]["allocation"] = 0.5
    with pytest.raises(ProjectValidationError, match="allocation"):
        validate_project(invalid, timeseries)


def test_calibration_uncertainty_and_mcda_interfaces(demo: tuple[dict, pd.DataFrame, object]) -> None:
    project, timeseries, _result = demo
    short_timeseries = timeseries.iloc[:60].copy()
    short_project = copy.deepcopy(project)
    short_project["simulation"]["end"] = str(short_timeseries["date"].iloc[-1].date())
    baseline = FullWaterMet2Model(short_project, short_timeseries).run()
    observed = baseline.component_daily[
        baseline.component_daily["component_id"] == "SEWER1"
    ].set_index("date")["outflow_ml"]
    calibration = grid_calibrate(
        short_project,
        short_timeseries,
        observed,
        {"components.SEWER1.release_a": [0.15, 0.20, 0.25]},
        "component_daily",
        "outflow_ml",
        "component_id=SEWER1",
    )
    assert calibration.trials.iloc[0]["components.SEWER1.release_a"] == pytest.approx(0.20)
    samples, percentiles = monte_carlo(
        short_project,
        short_timeseries,
        [
            {
                "path": "components.DM1.leakage_fraction",
                "distribution": "uniform",
                "low": 0.18,
                "high": 0.24,
            }
        ],
        samples=2,
        seed=1,
    )
    assert len(samples) == 2
    assert set(percentiles.columns) == {"kpi", "p05", "p50", "p95"}
    ranked = compromise_programming_rank(
        pd.DataFrame(
            {
                "alternative": ["A", "B"],
                "reliability": [0.95, 0.99],
                "cost": [10.0, 15.0],
            }
        ),
        {
            "reliability": {"goal": "max", "weight": 0.6},
            "cost": {"goal": "min", "weight": 0.4},
        },
    )
    assert set(ranked["rank"]) == {1, 2}
