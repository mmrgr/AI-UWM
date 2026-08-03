from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from watermet2_repro.analysis import grid_calibrate, pareto_grid_optimize
from watermet2_repro.full_engine import FullWaterMet2Model, load_project
from watermet2_repro.toolkit import WaterMet2Toolkit
from watermet2_repro.validation import prepare_project


PROJECT = Path("examples/demo_full/project.json")


def short_project(days: int = 10) -> tuple[dict, pd.DataFrame]:
    project, timeseries = load_project(PROJECT)
    frame = timeseries.iloc[:days].copy()
    project = copy.deepcopy(project)
    project["simulation"]["end"] = str(frame["date"].iloc[-1].date())
    project["pipeline_events"] = []
    project["interventions"] = []
    return project, frame


def test_explicit_hierarchy_and_edge_topology() -> None:
    project, timeseries = short_project()
    project.pop("supply_paths")
    project["supply_connections"] = []
    for suffix, allocation in (("1", 0.9), ("2", 0.1)):
        chain = [f"WR{suffix}", f"SC{suffix}", f"WTW{suffix}", f"TM{suffix}",
                 f"SR{suffix}", f"DM{suffix}"]
        project["supply_connections"].extend(
            {"from": source, "to": target, "allocation": 1.0}
            for source, target in zip(chain, chain[1:])
        )
        project["supply_connections"].append(
            {"from": f"DM{suffix}", "to": "LA1", "allocation": allocation}
        )
    project["indoor_areas"] = {
        "HOUSE": {
            "local_area": "LA1",
            "number_of_properties": 100,
            "occupancy_people_property": 2.5,
            "demand_profiles": [
                {"name": "bath", "base_value": 30, "unit": "l_capita_day"}
            ],
        }
    }
    prepared = prepare_project(project)
    assert np.isclose(sum(path["allocation"] for path in prepared["supply_paths"]), 1)
    result = FullWaterMet2Model(project, timeseries).run()
    assert set(result.subcatchment_daily["subcatchment_id"]) == {"SUB1"}
    assert set(result.indoor_daily["indoor_id"]) == {"HOUSE"}
    assert (result.indoor_daily["water_demand_ml"] > 0).all()


def test_official_climate_fields_and_selected_rwh_surface() -> None:
    project, timeseries = short_project(3)
    timeseries["rainfall_mm"] = [10.0, 10.0, 0.0]
    timeseries["temperature_c"] = [5.0, -2.0, 5.0]
    timeseries["precipitation_type"] = ["rain", "snow", "rain"]
    timeseries["snow_depth_mm"] = [0.0, 20.0, 10.0]
    timeseries["wind_m_s"] = 2.0
    timeseries["sunshine_h"] = 8.0
    timeseries["humidity_percent"] = 60.0
    area = project["local_areas"]["LA1"]
    area["latitude_deg"] = 59.91
    area["elevation_m"] = 50
    area["snow_gravity"] = 0.1
    area["weather_columns"].update({
        "precipitation_type": "precipitation_type",
        "snow_depth": "snow_depth_mm",
        "wind_speed": "wind_m_s",
        "sunshine_hours": "sunshine_h",
        "relative_humidity": "humidity_percent",
    })
    rwh = project["components"]["RWH_LOCAL"]
    rwh["start_date"] = "2011-01-01"
    rwh["source_surfaces"] = ["roof"]
    rwh["collection_fraction"] = 1.0
    result = FullWaterMet2Model(project, timeseries).run()
    rwh_rows = result.component_daily[result.component_daily.component_id == "RWH_LOCAL"]
    assert rwh_rows["inflow_ml"].sum() > 0
    assert result.area_daily.iloc[1]["runoff_ml"] == 0
    assert rwh_rows["inflow_ml"].sum() < result.area_daily["runoff_ml"].sum()


def test_transmission_sewer_has_no_storage_and_tracks_exfiltration_mass() -> None:
    project, timeseries = short_project(5)
    sewer = project["components"]["SEWER1"]
    sewer.update({
        "capacity_mode": "transmission",
        "daily_capacity_ml": 1.0,
        "infiltration_fraction": 0.03,
        "exfiltration_fraction": 0.02,
        "exfiltration_to": "RW1",
    })
    result = FullWaterMet2Model(project, timeseries).run()
    rows = result.component_daily[result.component_daily.component_id == "SEWER1"]
    assert (rows["storage_ml"] == 0).all()
    assert rows["cso_ml"].sum() > 0
    assert rows["exfiltration_ml"].sum() > 0
    exfiltration = result.pollutant_daily[
        (result.pollutant_daily.component_id == "SEWER1")
        & (result.pollutant_daily.stream == "exfiltration")
    ]
    assert exfiltration["mass_kg"].sum() > 0


def test_wtw_sludge_and_impact_breakdown() -> None:
    project, timeseries = short_project()
    wtw = project["components"]["WTW1"]
    wtw["raw_water_tss_mg_l"] = 15.0
    wtw["tss_removal_fraction"] = 0.9
    wtw["chemical_sludge_yield_kg_per_kg"] = {"alum": 0.26}
    result = FullWaterMet2Model(project, timeseries).run()
    rows = result.component_daily[result.component_daily.component_id == "WTW1"]
    assert rows["sludge_kg"].sum() > 0
    assert rows["electricity_ghg_kg_co2e"].sum() > 0
    assert rows["fossil_ghg_kg_co2e"].sum() > 0
    assert rows["embodied_ghg_kg_co2e"].sum() > 0


def test_annual_rehabilitation_selects_oldest_cohort_first() -> None:
    project, timeseries = load_project(PROJECT)
    frame = timeseries[
        (timeseries.date >= "2011-12-31") & (timeseries.date <= "2012-01-01")
    ].copy()
    project = copy.deepcopy(project)
    project["simulation"] = {"start": "2011-12-31", "end": "2012-01-01"}
    project["pipeline_events"] = []
    project["interventions"] = []
    project["pipelines"] = [{
        "id": "P",
        "component_id": "DM1",
        "length_m": 200,
        "diameter_mm": 300,
        "material": "grey_cast_iron",
        "cohorts": [
            {"length_m": 100, "age_years": 60, "material": "grey_cast_iron"},
            {"length_m": 100, "age_years": 20, "material": "grey_cast_iron"},
        ],
        "rehabilitation_method": "slip_lining_pe",
        "annual_rehabilitation_length_m": 50,
    }]
    model = FullWaterMet2Model(project, frame)
    result = model.run()
    cohorts = sorted(model.pipeline_state["P"]["cohorts"], key=lambda item: item["age_years"])
    assert [(item["length_m"], item["age_years"]) for item in cohorts] == [
        (50.0, 0.0), (100.0, 21.0), (50.0, 61.0)
    ]
    assert result.material_events.iloc[0]["trigger"] == "annual"


def test_toolkit_and_multiscale_aggregated_outputs(tmp_path: Path) -> None:
    project, timeseries = short_project()
    toolkit = WaterMet2Toolkit(project, timeseries)
    original = toolkit.get_input("components.DM1.leakage_fraction")
    toolkit.set_input("components.DM1.leakage_fraction", original - 0.01)
    result = toolkit.run()
    assert toolkit.list_components("wwtw") == ["WWTW1", "WWTW2"]
    assert not toolkit.get_result(
        "component_daily", component_id="DM1", columns=["date", "leakage_ml"]
    ).empty
    toolkit.write_results(tmp_path)
    for name in (
        "system_weekly.csv", "system_monthly.csv", "system_annual.csv",
        "subcatchment_monthly.csv", "component_annual.csv",
        "pollutant_monthly.csv", "recovery_annual.csv",
    ):
        assert (tmp_path / name).exists()
    assert not toolkit.get_result("system_daily", frequency="weekly").empty
    assert toolkit.get_result_value(
        "component_daily", "leakage_ml", component_id="DM1"
    ) >= 0
    clipped = toolkit.get_timeseries(
        ["rainfall_mm"], start=timeseries.iloc[1]["date"], end=timeseries.iloc[2]["date"]
    )
    assert len(clipped) == 2
    toolkit.set_timeseries_column("test_signal", 1.0)
    assert toolkit.get_timeseries(["test_signal"])["test_signal"].eq(1.0).all()
    assert len(result.subcatchment_daily) == len(timeseries)


def test_discrete_multiobjective_optimization() -> None:
    project, timeseries = short_project(5)
    trials = pareto_grid_optimize(
        project,
        timeseries,
        {"components.DM1.leakage_fraction": [0.10, 0.22]},
        {
            "mean_annual_leakage_ml": "min",
            "present_total_cost_eur": "min",
        },
    )
    assert len(trials) == 2
    assert trials["is_pareto"].any()


def test_groundwater_and_desalination_are_non_storage_resources() -> None:
    project, timeseries = short_project(3)
    resource = project["components"]["WR2"]
    resource["resource_type"] = "groundwater"
    resource.pop("inflow_column")
    resource.pop("initial_ml")
    resource.pop("capacity_ml")
    result = FullWaterMet2Model(project, timeseries).run()
    rows = result.component_daily[result.component_daily.component_id == "WR2"]
    assert rows["outflow_ml"].sum() > 0
    assert (rows["storage_ml"] == 0).all()
    np.testing.assert_allclose(rows["inflow_ml"], rows["outflow_ml"])


def test_pipeline_addition_and_retirement_material_flows() -> None:
    project, timeseries = short_project(3)
    project["pipelines"] = [{
        "id": "P",
        "component_id": "DM1",
        "length_m": 100,
        "diameter_mm": 300,
        "material": "grey_cast_iron",
        "age_years": 30,
    }]
    project["pipeline_events"] = [
        {
            "date": "2011-01-02", "pipeline_id": "P", "action": "add",
            "method": "rebuild_ductile_iron", "length_m": 10,
        },
        {
            "date": "2011-01-03", "pipeline_id": "P", "action": "retire",
            "length_m": 5,
        },
    ]
    model = FullWaterMet2Model(project, timeseries)
    result = model.run()
    assert np.isclose(model.pipeline_state["P"]["length_m"], 105)
    assert set(result.material_events["trigger"]) == {"addition", "retirement"}
    assert result.material_events["material_mass_kg"].min() < 0


def test_recharge_flooding_and_all_official_risk_codes() -> None:
    project, timeseries = short_project(3)
    timeseries["rainfall_mm"] = 40.0
    project["components"]["SEWER1"].update(
        {
            "capacity_mode": "transmission",
            "daily_capacity_ml": 0.1,
            "flood_depth_m": 0.2,
            "street_flow_width_m": 2.0,
            "dangerous_velocity_m_s": 0.0,
        }
    )
    result = FullWaterMet2Model(project, timeseries).run()
    assert result.area_daily["aquifer_recharge_ml"].sum() > 0
    assert result.flood_daily["flooded_area_m2"].sum() > 0
    assert result.flood_daily["high_velocity_flood_area_m2"].sum() > 0
    assert result.risk_summary["risk_code"].nunique() == 23
    assert set(result.risk_daily["risk_code"]) == {
        "R01HZ01", "R01HZ02", "R01HZ03", "R01HZ04", "R01HZ05",
        "R02HZ01", "R03HZ01", "R03HZ02", "R05HZ01", "R07HZ01",
        "R07HZ02", "R08HZ03", "R08HZ04", "R08HZ05", "R08HZ06",
        "R08HZ07", "R08HZ08", "R08HZ09", "R08HZ11", "R08HZ12",
        "R08HZ13", "R09HZ01", "R09HZ02",
    }


def test_asset_failure_regression_and_lifecycle_costs() -> None:
    project, timeseries = short_project(3)
    component = project["components"]["DM1"]
    component.update(
        {
            "capital_cost_eur": 36525.0,
            "investment_date": "2011-01-01",
            "lifetime_years": 10,
            "maintenance_fraction_capital_year": 0.01,
            "failure_model": {
                "model": "exponential",
                "intercept": -2.0,
                "age_coefficient": 0.03,
                "basis": "per_asset",
                "repair_cost_eur_failure": 1000.0,
                "duration_hours_failure": 8.0,
            },
        }
    )
    result = FullWaterMet2Model(project, timeseries).run()
    rows = result.component_daily[result.component_daily.component_id == "DM1"]
    assert rows["capital_cost_eur"].iloc[0] >= 36525.0
    assert rows["maintenance_cost_eur"].sum() > 0
    assert rows["annualized_capital_cost_eur"].sum() > 0
    assert rows["expected_failures"].sum() > 0
    assert rows["expected_outage_hours"].sum() > 0
    assert result.asset_daily["failure_repair_cost_eur"].sum() > 0


def test_appliance_energy_cost_and_import_export_accounting() -> None:
    project, timeseries = short_project(3)
    profile = project["local_areas"]["LA1"]["demand_profiles"][0]
    profile.update(
        {
            "electricity_kwh_m3": 2.0,
            "capital_cost_eur": 1000.0,
            "investment_date": "2011-01-01",
            "lifetime_years": 5,
            "maintenance_cost_eur_year": 365.25,
        }
    )
    resource = project["components"]["WR2"]
    resource["resource_type"] = "imported"
    resource.pop("inflow_column")
    resource["export_fraction"] = 0.1
    result = FullWaterMet2Model(project, timeseries).run()
    assert result.area_daily["appliance_electricity_kwh"].sum() > 0
    assert result.area_daily["appliance_capital_cost_eur"].sum() == 1000.0
    assert result.system_daily["imported_water_ml"].sum() > 0
    assert result.system_daily["exported_water_ml"].sum() > 0


def test_tap_quality_uses_treatment_asset_age_and_reservoir_storage() -> None:
    project, timeseries = short_project(2)
    project["components"]["WR1"]["raw_water_quality_index"] = 0.4
    project["components"]["WTW1"]["treated_water_quality_index"] = 0.95
    project["components"]["SR1"]["low_storage_quality_penalty"] = 0.2
    project["components"]["DM1"].update(
        {"asset_age_years": 50, "quality_decay_per_year": 0.01}
    )
    result = FullWaterMet2Model(project, timeseries).run()
    quality = result.system_daily["tap_water_quality_index"]
    assert quality.between(0, 1).all()
    assert (quality < 0.95).all()
    assert set(result.system_daily["tap_water_quality_class"]) <= {
        "excellent", "good", "acceptable", "poor"
    }
    risk = result.risk_daily[result.risk_daily.risk_code == "R05HZ01"]
    np.testing.assert_allclose(risk["value"], 1.0 - quality)


def test_calibration_supports_monthly_and_yearly_scales() -> None:
    project, timeseries = short_project(40)
    baseline = FullWaterMet2Model(project, timeseries).run()
    observed = baseline.system_daily.set_index("date")["water_demand_ml"]
    calibrated = grid_calibrate(
        project,
        timeseries,
        observed,
        {"components.DM1.leakage_fraction": [0.2, 0.22]},
        "system_daily",
        "water_demand_ml",
        frequency="MS",
        aggregation="sum",
    )
    assert np.isclose(calibrated.trials.iloc[0]["nse"], 1.0, equal_nan=True)


def test_wwtw_sludge_digestion_dewatering_drying_and_end_use() -> None:
    project, timeseries = short_project(3)
    project["components"]["WWTW1"]["sludge_process"] = {
        "digestion_mass_reduction_fraction": 0.4,
        "dewatered_dry_solids_fraction": 0.2,
        "dried_dry_solids_fraction": 0.9,
        "end_use_fraction": 0.8,
    }
    result = FullWaterMet2Model(project, timeseries).run()
    recovery = result.recovery_daily[
        result.recovery_daily.component_id == "WWTW1"
    ].groupby("product")["amount"].sum()
    assert {
        "digested_sludge_dry_solids",
        "dewatered_sludge",
        "dried_sludge",
        "biosolids_to_end_use",
    } <= set(recovery.index)
    assert recovery["dewatered_sludge"] > recovery["dried_sludge"]
    assert recovery["biosolids_to_end_use"] < recovery["digested_sludge_dry_solids"]
