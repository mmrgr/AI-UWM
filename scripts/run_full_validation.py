from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from watermet2_repro.analysis import (
    compromise_programming_rank,
    grid_calibrate,
    monte_carlo,
    pareto_grid_optimize,
    summarize_kpis,
)
from watermet2_repro.full_engine import FullWaterMet2Model, load_project
from watermet2_repro.toolkit import WaterMet2Toolkit
from watermet2_repro.validation import ProjectValidationError, validate_project


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation_artifacts"
SEED = 3302


def set_path(root: dict[str, Any], path: str, value: Any) -> None:
    target: Any = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def generate_timeseries() -> pd.DataFrame:
    dates = pd.date_range("2011-01-01", "2012-12-31", freq="D")
    rng = np.random.default_rng(SEED)
    day = dates.dayofyear.to_numpy()
    temperature = (
        6.5
        + 10.5 * np.sin(2 * np.pi * (day - 172) / 365.25)
        + rng.normal(0, 3, len(dates))
    )
    wet = rng.random(len(dates)) < 0.45
    rainfall = np.where(wet, rng.gamma(1.25, 4.2, len(dates)), 0.0)
    precipitation_type = np.where(
        temperature <= 0, "snow", np.where(temperature < 2, "sleet", "rain")
    )
    snow_depth = np.zeros(len(dates))
    for index in range(1, len(dates)):
        accumulation = rainfall[index] if precipitation_type[index] == "snow" else 0.0
        melt = max(0.0, temperature[index]) * 1.5
        snow_depth[index] = max(0.0, snow_depth[index - 1] + accumulation - melt)
    seasonal_inflow = 1 + 0.35 * np.sin(2 * np.pi * (day - 105) / 365.25)
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_c": temperature,
            "rainfall_mm": rainfall,
            "precipitation_type": precipitation_type,
            "snow_depth_mm": snow_depth,
            "wind_m_s": np.maximum(0.1, rng.normal(2.5, 0.7, len(dates))),
            "sunshine_h": np.clip(
                6 + 4 * np.sin(2 * np.pi * (day - 172) / 365.25), 0, 18
            ),
            "humidity_percent": np.clip(rng.normal(72, 8, len(dates)), 35, 100),
            "wr1_inflow_ml": np.maximum(0, (287_000 / 365.25) * seasonal_inflow),
            "wr2_inflow_ml": np.maximum(0, (12_000 / 365.25) * seasonal_inflow),
        }
    )


def build_validation_project() -> dict[str, Any]:
    project = json.loads(
        (ROOT / "examples" / "demo_full" / "project.json").read_text(encoding="utf-8")
    )
    project["name"] = "WaterMet2 synthetic full-flow validation"
    project["database_file"] = str((ROOT / "data" / "watermet2_database.json").resolve())
    project["timeseries_file"] = "synthetic_timeseries.csv"
    project["components"]["SEWER1"]["infiltration_fraction"] = 0.03
    project["components"]["SEWER1"]["exfiltration_fraction"] = 0.01

    # Exercise all documented energy carriers and chemicals through the same generic engines.
    database = json.loads(
        (ROOT / "data" / "watermet2_database.json").read_text(encoding="utf-8")
    )
    fuels = {
        name: 0.00001
        for name in database["energy_sources"]
        if name not in {"electricity", "diesel"}
    }
    project["components"]["WTW2"]["fossil_fuels"].update(fuels)
    doses = project["components"]["WTW1"]["chemical_doses_kg_m3"]
    for chemical in database["chemicals"]:
        doses.setdefault(chemical, 0.00001)

    # Add a second Local area with separate sanitary/storm drainage and property-based population.
    area2 = copy.deepcopy(project["local_areas"]["LA1"])
    area2.pop("base_population")
    area2["subcatchment"] = "SUB2"
    area2["number_of_properties"] = 8_000
    area2["occupancy_people_property"] = 2.5
    area2["annual_population_growth"] = 0.005
    area2["area_ha"] = 800
    area2.pop("combined_sewer")
    area2["sanitary_sewer"] = "SAN_SEWER"
    area2["storm_sewer"] = "STORM_SEWER"
    for profile in area2["demand_profiles"]:
        if profile["unit"] == "ml_day":
            profile["base_value"] *= 0.03
    indoor_profiles = [
        profile for profile in area2["demand_profiles"]
        if profile["unit"] == "l_capita_day"
    ]
    area2["demand_profiles"] = [
        profile for profile in area2["demand_profiles"]
        if profile["unit"] != "l_capita_day"
    ]
    project["local_areas"]["LA2"] = area2
    project["subcatchments"] = {
        "SUB1": {"local_areas": ["LA1"]},
        "SUB2": {"local_areas": ["LA2"]},
    }
    project["indoor_areas"] = {
        "HOUSE_LA2": {
            "local_area": "LA2",
            "number_of_properties": 8_000,
            "occupancy_people_property": 2.5,
            "annual_population_growth": 0.005,
            "demand_profiles": indoor_profiles,
        }
    }
    project["components"]["SAN_SEWER"] = {
        "kind": "sewer",
        "sewer_type": "sanitary",
        "capacity_mode": "transmission",
        "capacity_ml": 80,
        "daily_capacity_ml": 80,
        "infiltration_fraction": 0.02,
        "exfiltration_fraction": 0.005,
        "overflow_to": "RW1",
        "electricity_kwh_m3": 0.01,
    }
    project["components"]["STORM_SEWER"] = {
        "kind": "sewer",
        "sewer_type": "storm",
        "capacity_mode": "transmission",
        "capacity_ml": 150,
        "daily_capacity_ml": 5,
        "overflow_to": "RW1",
        "electricity_kwh_m3": 0.01,
        "flood_depth_m": 0.12,
        "street_flow_width_m": 4.0,
        "dangerous_velocity_m_s": 0.0,
    }
    for source, target in (("SAN_SEWER", "WWTW1"), ("STORM_SEWER", "WWTW2")):
        project["wastewater_connections"].append(
            {"from": source, "to": target, "fraction": 1.0}
        )
    for path in copy.deepcopy(project["supply_paths"]):
        path["id"] += "_LA2"
        path["local_area"] = "LA2"
        project["supply_paths"].append(path)
    project["supply_connections"] = []
    for suffix, allocation in (("1", 0.90), ("2", 0.10)):
        chain = [
            f"WR{suffix}", f"SC{suffix}", f"WTW{suffix}",
            f"TM{suffix}", f"SR{suffix}", f"DM{suffix}",
        ]
        project["supply_connections"].extend(
            {"from": source, "to": target, "allocation": 1.0}
            for source, target in zip(chain, chain[1:])
        )
        for area_id in ("LA1", "LA2"):
            project["supply_connections"].append({
                "from": f"DM{suffix}", "to": area_id, "allocation": allocation
            })

    # COD demonstrates that pollutant names are not hard-coded.
    for area in project["local_areas"].values():
        area["sanitary_pollutant_load_kg_capita_day"]["COD"] = 0.10
        for surface in area["surfaces"].values():
            surface["pollutant_emc_mg_l"]["COD"] = 30.0
    for component in project["components"].values():
        if component["kind"] in {"reuse", "wwtw"}:
            component.setdefault("pollutant_removal_fraction", {})["COD"] = 0.75
    for area in project["local_areas"].values():
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
    project["components"]["RWH_LOCAL"]["source_surfaces"] = ["roof", "road_pavement"]
    project["components"]["WTW1"].update({
        "raw_water_tss_mg_l": 15.0,
        "tss_removal_fraction": 0.9,
        "chemical_sludge_yield_kg_per_kg": {"alum": 0.26},
    })
    project["components"]["WWTW2"]["process_recovery"] = {
        "biogas_kg_per_kg_sludge": 0.05,
        "biogas_lhv_kwh_kg": 6.0,
        "chp_electric_efficiency": 0.35,
        "chp_heat_efficiency": 0.45,
        "effluent_temperature_drop_c": 2.0,
        "heat_recovery_efficiency": 0.6,
    }
    project["components"]["WWTW1"]["sludge_process"] = {
        "digestion_mass_reduction_fraction": 0.4,
        "dewatered_dry_solids_fraction": 0.2,
        "dried_dry_solids_fraction": 0.9,
        "end_use_fraction": 0.8,
    }
    project["components"]["TM1"]["hydraulic_head_m"] = 5.0
    project["components"]["TM1"]["turbine_efficiency"] = 0.75
    project["components"]["DM1"].update({
        "capital_cost_eur": 25_000_000,
        "investment_date": "2011-01-01",
        "lifetime_years": 50,
        "maintenance_fraction_capital_year": 0.01,
    })
    project["components"]["SEWER1"].update({
        "flood_depth_m": 0.15,
        "street_flow_width_m": 6.0,
        "dangerous_velocity_m_s": 0.3,
    })
    for profile in indoor_profiles:
        profile["electricity_kwh_m3"] = 0.2
        profile["maintenance_cost_eur_year"] = 10.0
    project["risk_settings"] = {
        "baseline_recharge_ml_day": 50.0,
        "renewable_target_kwh_day": 1000.0,
        "scarcity_cost_eur_ml": 1000.0,
        "energy_price_stress_eur_kwh": 0.05,
        "injury_exposure_per_m2": 0.000001,
    }

    # Exercise every rehabilitation method and every diameter class.
    project["pipelines"] = [
        {"id": "P1", "component_id": "DM1", "length_m": 100_000, "diameter_mm": 300,
         "material": "grey_cast_iron", "age_years": 60, "leakage_growth_per_year": 0.01,
         "maintenance_cost_eur_m_year": 0.5,
         "failure_model": {
             "model": "exponential", "intercept": -2.5,
             "age_coefficient": 0.035, "basis": "per_km",
             "repair_cost_eur_failure": 4000,
             "duration_hours_failure": 6,
         }},
        {"id": "P2", "component_id": "DM2", "length_m": 20_000, "diameter_mm": 150,
         "material": "PVC", "age_years": 30,
         "rehabilitation_method": "polyurethane_lining",
         "annual_rehabilitation_rate": 0.002},
        {"id": "P3", "component_id": "TM1", "length_m": 30_000, "diameter_mm": 700,
         "material": "ductile_iron", "age_years": 40},
        {"id": "P4", "component_id": "TM2", "length_m": 10_000, "diameter_mm": 350,
         "material": "mild_steel", "age_years": 25},
    ]
    project["pipeline_events"] = [
        {"date": "2012-01-01", "pipeline_id": "P1", "method": "slip_lining_pe",
         "length_m": 1_000, "new_leakage_fraction": 0.20},
        {"date": "2012-04-01", "pipeline_id": "P2", "method": "polyurethane_lining",
         "length_m": 200},
        {"date": "2012-07-01", "pipeline_id": "P3", "method": "pipe_cracking_lining",
         "length_m": 300},
        {"date": "2012-10-01", "pipeline_id": "P4", "method": "rebuild_ductile_iron",
         "length_m": 100},
    ]
    project["interventions"].append(
        {
            "id": "wtw1_energy_change",
            "date": "2012-01-01",
            "set": [{"path": "components.WTW1.electricity_kwh_m3", "value": 0.50}],
        }
    )
    return project


def check_record(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_project = build_validation_project()
    timeseries = generate_timeseries()
    (OUTPUT / "synthetic_project.json").write_text(
        json.dumps(raw_project, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    timeseries.to_csv(OUTPUT / "synthetic_timeseries.csv", index=False)
    project, loaded_timeseries = load_project(OUTPUT / "synthetic_project.json")
    model = FullWaterMet2Model(project, loaded_timeseries)
    result = model.run()
    result.write(OUTPUT / "baseline")
    checks: list[dict[str, Any]] = []

    expected_kinds = {
        "water_resource", "supply_conduit", "wtw", "trunk_main",
        "service_reservoir", "distribution_main", "reuse", "sewer",
        "wwtw", "receiving_water",
    }
    actual_kinds = set(result.component_daily["kind"])
    checks.append(check_record(
        "全部组件族被执行", expected_kinds <= actual_kinds,
        f"执行 {len(actual_kinds)} 类组件、{result.component_daily.component_id.nunique()} 个组件",
    ))
    reuse_scales = {
        component["scale"] for component in project["components"].values()
        if component["kind"] == "reuse"
    }
    sewer_types = {
        component["sewer_type"] for component in project["components"].values()
        if component["kind"] == "sewer"
    }
    checks.append(check_record(
        "回用尺度与排水制式覆盖",
        reuse_scales == {"local", "subcatchment", "system"}
        and sewer_types == {"combined", "sanitary", "storm"},
        f"reuse={sorted(reuse_scales)}, sewer={sorted(sewer_types)}",
    ))
    checks.append(check_record(
        "四级层级与逐边供水拓扑",
        not result.indoor_daily.empty
        and result.subcatchment_daily.subcatchment_id.nunique() == 2
        and len(project["supply_paths"]) == 4,
        f"Indoor={result.indoor_daily.indoor_id.nunique()}, Subcatchment={result.subcatchment_daily.subcatchment_id.nunique()}, 自动路径={len(project['supply_paths'])}",
    ))
    activity = result.component_daily.groupby("component_id")["inflow_ml"].sum()
    checks.append(check_record(
        "所有组件均有实际活动", bool((activity > 0).all()),
        f"最小累计入流={activity.min():.6f} ML",
    ))
    rwh_inflow = result.component_daily[
        result.component_daily.component_id == "RWH_LOCAL"
    ].inflow_ml.sum()
    checks.append(check_record(
        "官方气象字段与指定集雨面被执行",
        {"rain", "sleet", "snow"} <= set(loaded_timeseries.precipitation_type)
        and rwh_inflow > 0,
        f"降水类型=3, RWH收集={rwh_inflow:.1f} ML",
    ))

    demand_error = (
        result.area_daily["water_demand_ml"]
        - result.area_daily["delivered_total_ml"]
        - result.area_daily["unmet_ml"]
    ).abs().max()
    checks.append(check_record(
        "逐日需求质量守恒", demand_error < 1e-9,
        f"最大绝对残差={demand_error:.3e} ML",
    ))
    resource_errors = []
    for resource_id in ("WR1", "WR2"):
        component = project["components"][resource_id]
        rows = result.component_daily[result.component_daily.component_id == resource_id]
        residual = (
            component["initial_ml"] + rows["inflow_ml"].sum()
            - rows["outflow_ml"].sum() - rows["loss_ml"].sum()
            - rows["overflow_ml"].sum() - rows.iloc[-1]["storage_ml"]
        )
        resource_errors.append(abs(residual))
    checks.append(check_record(
        "水源跨期质量守恒", max(resource_errors) < 1e-8,
        f"最大绝对残差={max(resource_errors):.3e} ML",
    ))
    storage_errors = []
    for component_id, initial in model.storage.items():
        if project["components"][component_id]["kind"] not in {"sewer", "wwtw", "reuse"}:
            continue
        rows = result.component_daily[result.component_daily.component_id == component_id]
        starting = float(project["components"][component_id].get("initial_ml", 0))
        residual = (
            starting + rows["inflow_ml"].sum() - rows["outflow_ml"].sum()
            - rows["loss_ml"].sum() - rows["overflow_ml"].sum()
            - rows.iloc[-1]["storage_ml"]
        )
        storage_errors.append(abs(residual))
    checks.append(check_record(
        "排水/处理/回用储量守恒", max(storage_errors) < 1e-8,
        f"最大绝对残差={max(storage_errors):.3e} ML",
    ))

    component = result.component_daily
    impact_errors = []
    for caused, avoided, net in (
        ("ghg_caused_kg_co2e", "ghg_avoided_kg_co2e", "ghg_net_kg_co2e"),
        ("acidification_caused_kg_so2e", "acidification_avoided_kg_so2e",
         "acidification_net_kg_so2e"),
        ("eutrophication_caused_kg_po4e", "eutrophication_avoided_kg_po4e",
         "eutrophication_net_kg_po4e"),
    ):
        impact_errors.append((component[caused] - component[avoided] - component[net]).abs().max())
    checks.append(check_record(
        "环境影响净值恒等式", max(impact_errors) < 1e-8,
        f"最大绝对残差={max(impact_errors):.3e}",
    ))

    wtw1 = component[component.component_id == "WTW1"]
    before = wtw1[wtw1.date < "2012-01-01"]
    after = wtw1[wtw1.date >= "2012-01-01"]
    before_rate = before.electricity_kwh.sum() / (before.treated_ml.sum() * 1000)
    after_rate = after.electricity_kwh.sum() / (after.treated_ml.sum() * 1000)
    checks.append(check_record(
        "分期干预按日期生效",
        abs(before_rate - 0.343) < 1e-12 and abs(after_rate - 0.50) < 1e-12,
        f"干预前={before_rate:.3f}, 干预后={after_rate:.3f} kWh/m3",
    ))
    expected_methods = set(project["rehabilitation_methods"])
    actual_methods = set(result.material_events["method"])
    checks.append(check_record(
        "全部管网修复方法被执行",
        actual_methods == expected_methods and (result.material_events.capital_cost_eur > 0).all(),
        f"执行方法={sorted(actual_methods)}",
    ))
    annual_events = result.material_events[
        result.material_events["trigger"] == "annual"
    ]
    checks.append(check_record(
        "年度修复率与oldest-first资产队列",
        not annual_events.empty and len(model.pipeline_state["P2"]["cohorts"]) >= 2,
        f"自动修复事件={len(annual_events)}, P2管龄组={len(model.pipeline_state['P2']['cohorts'])}",
    ))

    pollutant_names = set(result.pollutant_daily["pollutant"])
    recovery_names = set(result.recovery_daily["product"])
    chemical_columns = {
        f"chemical_{name}_kg" for name in project["chemicals"]
    }
    used_chemicals = {
        column for column in chemical_columns
        if column in component and component[column].sum() > 0
    }
    checks.append(check_record(
        "污染物/化学品/资源回收覆盖",
        {"BOD", "TSS", "TN", "TP", "COD"} <= pollutant_names
        and used_chemicals == chemical_columns
        and {"biogas", "ammonium_nitrate", "single_superphosphate", "urea",
             "generated_electricity", "recovered_heat",
             "digested_sludge_dry_solids", "dewatered_sludge",
             "dried_sludge", "biosolids_to_end_use"} <= recovery_names,
        f"污染物={sorted(pollutant_names)}, 化学品={len(used_chemicals)}, 回收品={len(recovery_names)}",
    ))
    sewer_streams = set(
        result.pollutant_daily[
            result.pollutant_daily.component_id == "SEWER1"
        ].stream
    )
    checks.append(check_record(
        "WTW污泥及CSO/外渗污染物",
        wtw1.sludge_kg.sum() > 0 and {"cso", "exfiltration"} <= sewer_streams,
        f"WTW污泥={wtw1.sludge_kg.sum():.1f} kg, 排水污染物流={sorted(sewer_streams)}",
    ))
    gwr_pollutants = result.pollutant_daily[
        result.pollutant_daily.component_id == "GWR_SUB"
    ].groupby(["pollutant", "stream"]).mass_kg.sum().unstack(fill_value=0)
    removal_errors = []
    for pollutant, expected in project["components"]["GWR_SUB"]["pollutant_removal_fraction"].items():
        removed = gwr_pollutants.loc[pollutant, "removed_to_sludge"]
        discharged = gwr_pollutants.loc[pollutant, "recycled_outflow"]
        removal_errors.append(abs(removed / (removed + discharged) - expected))
    checks.append(check_record(
        "回用污染物去除率正确", max(removal_errors) < 1e-12,
        f"最大比例残差={max(removal_errors):.3e}",
    ))
    tm1 = component[component.component_id == "TM1"]
    generation_rate = (
        project["components"]["TM1"]["energy_generation_kwh_m3"]
        + 9.81 * project["components"]["TM1"]["hydraulic_head_m"]
        * project["components"]["TM1"]["turbine_efficiency"] / 3600.0
    )
    generation_error = (
        tm1.energy_generated_kwh - tm1.inflow_ml * 1000
        * generation_rate
    ).abs().max()
    checks.append(check_record(
        "发电量计算正确", generation_error < 1e-9,
        f"最大绝对残差={generation_error:.3e} kWh",
    ))

    checks.append(check_record(
        "地下水补给与洪涝面积/流速模型",
        result.area_daily.aquifer_recharge_ml.sum() > 0
        and result.flood_daily.flooded_area_m2.sum() > 0
        and result.flood_daily.high_velocity_flood_area_m2.sum() > 0,
        f"补给={result.area_daily.aquifer_recharge_ml.sum():.1f} ML，"
        f"淹没面积日={result.flood_daily.flooded_area_m2.sum():.1f} m2-day",
    ))
    checks.append(check_record(
        "23项官方风险代码完整计算",
        result.risk_summary.risk_code.nunique() == 23
        and len(result.risk_daily) == 23 * len(loaded_timeseries),
        f"风险代码={result.risk_summary.risk_code.nunique()}，"
        f"逐日记录={len(result.risk_daily)}",
    ))
    checks.append(check_record(
        "资产失效与生命周期成本",
        result.asset_daily.expected_failures.sum() > 0
        and result.asset_daily.failure_repair_cost_eur.sum() > 0
        and component.maintenance_cost_eur.sum() > 0
        and component.annualized_capital_cost_eur.sum() > 0,
        f"期望失效={result.asset_daily.expected_failures.sum():.3f}，"
        f"维护成本={component.maintenance_cost_eur.sum():.1f} EUR",
    ))
    checks.append(check_record(
        "用水设备能耗与运维成本",
        result.area_daily.appliance_electricity_kwh.sum() > 0
        and result.area_daily.appliance_operational_cost_eur.sum() > 0,
        f"设备电耗={result.area_daily.appliance_electricity_kwh.sum():.1f} kWh，"
        f"设备运维={result.area_daily.appliance_operational_cost_eur.sum():.1f} EUR",
    ))

    invalid_allocation = copy.deepcopy(project)
    invalid_allocation["supply_connections"][-1]["allocation"] = 0.5
    allocation_rejected = False
    try:
        validate_project(invalid_allocation, loaded_timeseries)
    except ProjectValidationError:
        allocation_rejected = True
    missing_day_rejected = False
    try:
        validate_project(project, loaded_timeseries.drop(index=10))
    except ProjectValidationError:
        missing_day_rejected = True
    checks.append(check_record(
        "错误输入被拒绝", allocation_rejected and missing_day_rejected,
        f"错误分配={allocation_rejected}, 缺日={missing_day_rejected}",
    ))

    calibration_days = 180
    calibration_project = copy.deepcopy(project)
    calibration_timeseries = loaded_timeseries.iloc[:calibration_days].copy()
    calibration_project["simulation"]["end"] = str(
        calibration_timeseries.date.iloc[-1].date()
    )
    observed = result.component_daily[
        result.component_daily.component_id == "SEWER1"
    ].iloc[:calibration_days].set_index("date")["outflow_ml"]
    observed.rename("observed_sewer_outflow_ml").to_csv(OUTPUT / "synthetic_observed.csv")
    calibration = grid_calibrate(
        calibration_project,
        calibration_timeseries,
        observed,
        {
            "components.SEWER1.release_a": [0.12, 0.20, 0.28],
            "local_areas.LA1.surfaces.pervious.runoff_coefficient": [0.08, 0.14, 0.20],
        },
        "component_daily",
        "outflow_ml",
        "component_id=SEWER1",
    )
    calibration.trials.to_csv(OUTPUT / "calibration_trials.csv", index=False)
    best = calibration.trials.iloc[0]
    calibration_ok = (
        abs(best["components.SEWER1.release_a"] - 0.20) < 1e-12
        and abs(best["local_areas.LA1.surfaces.pervious.runoff_coefficient"] - 0.14) < 1e-12
        and abs(best["nse"] - 1.0) < 1e-12
    )
    checks.append(check_record(
        "校准找回已知真值", calibration_ok,
        f"a={best['components.SEWER1.release_a']:.2f}, runoff={best['local_areas.LA1.surfaces.pervious.runoff_coefficient']:.2f}, NSE={best['nse']:.6f}",
    ))

    uncertainty_spec = [
        {"path": "components.DM1.leakage_fraction", "distribution": "uniform",
         "low": 0.15, "high": 0.25},
        {"path": "local_areas.LA1.surfaces.pervious.runoff_coefficient",
         "distribution": "triangular", "low": 0.08, "mode": 0.14, "high": 0.22},
    ]
    samples1, percentiles = monte_carlo(
        project, loaded_timeseries, uncertainty_spec, samples=12, seed=SEED
    )
    samples2, _ = monte_carlo(
        project, loaded_timeseries, uncertainty_spec, samples=12, seed=SEED
    )
    samples1.to_csv(OUTPUT / "uncertainty_samples.csv", index=False)
    percentiles.to_csv(OUTPUT / "uncertainty_percentiles.csv", index=False)
    checks.append(check_record(
        "不确定性分析可复现且分位数有序",
        samples1.equals(samples2)
        and bool((percentiles.p05 <= percentiles.p50).all())
        and bool((percentiles.p50 <= percentiles.p95).all()),
        f"样本数={len(samples1)}, seed={SEED}",
    ))

    scenarios = []
    for name, changes in (
        ("BAU", {}),
        ("Leakage reduction", {"components.DM1.leakage_fraction": 0.12}),
        ("Enhanced reuse", {
            "components.RWH_LOCAL.collection_fraction": 0.85,
            "components.GWR_SUB.collection_fraction": 0.85,
        }),
    ):
        scenario = copy.deepcopy(project)
        for path, value in changes.items():
            set_path(scenario, path, value)
        scenario_result = FullWaterMet2Model(scenario, loaded_timeseries).run()
        record = {"alternative": name, **summarize_kpis(scenario_result)}
        record["potable_delivered_ml"] = scenario_result.area_daily.potable_delivered_ml.sum()
        scenarios.append(record)
    drought_project = copy.deepcopy(project)
    drought_project["simulation"]["end"] = "2011-03-01"
    drought_timeseries = loaded_timeseries.iloc[:60].copy()
    drought_timeseries[["wr1_inflow_ml", "wr2_inflow_ml"]] = 0.0
    drought_project["components"]["WR1"]["initial_ml"] = 0.0
    drought_project["components"]["WR2"]["initial_ml"] = 0.0
    drought = FullWaterMet2Model(drought_project, drought_timeseries).run()
    scenario_frame = pd.DataFrame(scenarios)
    scenario_frame.to_csv(OUTPUT / "alternatives.csv", index=False)
    bau = scenario_frame.set_index("alternative").loc["BAU"]
    leakage = scenario_frame.set_index("alternative").loc["Leakage reduction"]
    reuse = scenario_frame.set_index("alternative").loc["Enhanced reuse"]
    checks.append(check_record(
        "情景方向与极端缺水响应正确",
        leakage.mean_annual_leakage_ml < bau.mean_annual_leakage_ml
        and reuse.potable_delivered_ml < bau.potable_delivered_ml
        and drought.system_daily.unmet_demand_ml.sum() > 0,
        f"漏损 {bau.mean_annual_leakage_ml:.1f}->{leakage.mean_annual_leakage_ml:.1f} ML/yr；干旱缺水={drought.system_daily.unmet_demand_ml.sum():.1f} ML",
    ))
    criteria = {
        "reliability_fraction": {"goal": "max", "weight": 0.35},
        "mean_annual_ghg_net_kg_co2e": {"goal": "min", "weight": 0.25},
        "mean_annual_eutrophication_net_kg_po4e": {"goal": "min", "weight": 0.15},
        "present_total_cost_eur": {"goal": "min", "weight": 0.25},
    }
    ranked = compromise_programming_rank(scenario_frame, criteria)
    ranked.to_csv(OUTPUT / "ranking.csv", index=False)
    ranks = sorted(ranked["rank"].tolist())
    checks.append(check_record(
        "多准则排序完整且距离单调", ranks == [1, 2, 3]
        and ranked.compromise_distance.is_monotonic_increasing,
        f"排序={' > '.join(ranked.alternative.tolist())}",
    ))
    toolkit_project = copy.deepcopy(project)
    toolkit_timeseries = loaded_timeseries.iloc[:10].copy()
    toolkit_project["simulation"]["end"] = str(
        toolkit_timeseries.date.iloc[-1].date()
    )
    toolkit = WaterMet2Toolkit(toolkit_project, toolkit_timeseries)
    toolkit.run()
    checks.append(check_record(
        "Python Toolkit读写运行检索",
        toolkit.get_input("components.DM1.kind") == "distribution_main"
        and not toolkit.get_result("component_daily", component_id="DM1").empty
        and "subcatchment_daily" in toolkit.list_result_tables(),
        f"结果表={len(toolkit.list_result_tables())}, 组件={len(toolkit.list_components())}",
    ))
    optimization_project = copy.deepcopy(project)
    optimization_timeseries = loaded_timeseries.iloc[:30].copy()
    optimization_project["simulation"]["end"] = str(
        optimization_timeseries.date.iloc[-1].date()
    )
    pareto = pareto_grid_optimize(
        optimization_project,
        optimization_timeseries,
        {"components.DM1.leakage_fraction": [0.12, 0.22]},
        {
            "mean_annual_leakage_ml": "min",
            "present_total_cost_eur": "min",
        },
    )
    pareto.to_csv(OUTPUT / "pareto_trials.csv", index=False)
    checks.append(check_record(
        "多目标Pareto优化",
        len(pareto) == 2 and pareto.is_pareto.any(),
        f"试验={len(pareto)}, Pareto解={int(pareto.is_pareto.sum())}",
    ))

    summary = {
        "seed": SEED,
        "simulation_days": len(loaded_timeseries),
        "components": len(project["components"]),
        "local_areas": len(project["local_areas"]),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "baseline_kpis": summarize_kpis(result),
    }
    return checks, summary


def write_report(checks: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    payload = {"summary": summary, "checks": checks}
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status = "通过" if summary["checks_passed"] == summary["checks_total"] else "失败"
    lines = [
        "# WaterMet² 合成数据全流程验证报告",
        "",
        f"- 总体结果：**{status}**",
        f"- 随机种子：`{summary['seed']}`",
        f"- 模拟长度：{summary['simulation_days']} 天",
        f"- 系统规模：{summary['components']} 个组件、{summary['local_areas']} 个 Local area",
        f"- 验收项：{summary['checks_passed']}/{summary['checks_total']} 通过",
        "",
        "| 验收项 | 结果 | 证据 |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(
            f"| {item['name']} | {'通过' if item['passed'] else '失败'} | {item['detail']} |"
        )
    lines.extend([
        "",
        "## 说明",
        "",
        "该报告验证代码实现与内部数学约束的一致性，包括已知真值回标、质量守恒和确定性复现。"
        "它不能替代使用真实城市数据进行的外部效度验证。",
    ])
    (OUTPUT / "VALIDATION_REPORT_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    checks, summary = run_validation()
    write_report(checks, summary)
    print(
        f"全流程验证：{summary['checks_passed']}/{summary['checks_total']} 项通过；"
        f"报告：{OUTPUT / 'VALIDATION_REPORT_CN.md'}"
    )
    if summary["checks_passed"] != summary["checks_total"]:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise SystemExit(f"验证失败：{failed}")


if __name__ == "__main__":
    main()
