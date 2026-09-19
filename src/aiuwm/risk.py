from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .full_engine import FullModelResult


RISK_CATALOG = {
    "R01HZ01": ("resource_cost_exposure_eur", "EUR/day", "Reduced resource availability"),
    "R01HZ02": ("regulatory_investment_eur", "EUR/day", "Required sustainability investment"),
    "R01HZ03": ("asset_renewal_exposure_eur", "EUR/day", "Ageing asset renewal"),
    "R01HZ04": ("capacity_shortfall_ml", "ML/day", "Demand exceeds capacity"),
    "R01HZ05": ("energy_cost_exposure_eur", "EUR/day", "Energy cost increase"),
    "R02HZ01": ("renewable_energy_shortfall_kwh", "kWh/day", "Renewable production shortfall"),
    "R03HZ01": ("cso_volume_ml", "ML/day", "Combined sewer overflow"),
    "R03HZ02": ("untreated_wastewater_ml", "ML/day", "Insufficient wastewater treatment"),
    "R05HZ01": ("tap_quality_proxy", "index", "Low tap-water quality"),
    "R07HZ01": ("flooded_area_m2", "m2/day", "Urban flooding"),
    "R07HZ02": ("flooded_area_m2", "m2/day", "Property flooding"),
    "R08HZ03": ("industrial_unmet_ml", "ML/day", "Industrial demand shortfall"),
    "R08HZ04": ("supply_outage_hours", "hours/day", "Prolonged supply interruption"),
    "R08HZ05": ("per_capita_demand_l_day", "L/capita/day", "High-consumption habits"),
    "R08HZ06": ("expected_pipe_failures", "failures/day", "Pipe bursts and outages"),
    "R08HZ07": ("storage_deficit_fraction", "fraction", "Insufficient storage"),
    "R08HZ08": ("storage_deficit_fraction", "fraction", "Reservoir under-capacity"),
    "R08HZ09": ("recharge_deficit_ml", "ML/day", "Reduced aquifer recharge"),
    "R08HZ11": ("low_pressure_proxy_ml", "ML/day", "Low pressure"),
    "R08HZ12": ("temperature_demand_excess_ml", "ML/day", "Temperature-driven demand"),
    "R08HZ13": ("drought_shortfall_ml", "ML/day", "Drought supply shortfall"),
    "R09HZ01": ("injury_exposure_index", "index", "Public injury exposure"),
    "R09HZ02": ("high_velocity_flood_area_m2", "m2/day", "High-velocity street runoff"),
}

RISK_METHODS = {
    "R01HZ01": "derived_physical_indicator",
    "R01HZ02": "direct_model_output",
    "R01HZ03": "derived_physical_indicator",
    "R01HZ04": "direct_model_output",
    "R01HZ05": "derived_physical_indicator",
    "R02HZ01": "derived_physical_indicator",
    "R03HZ01": "direct_model_output",
    "R03HZ02": "direct_model_output",
    "R05HZ01": "deterministic_proxy",
    "R07HZ01": "derived_physical_indicator",
    "R07HZ02": "deterministic_proxy",
    "R08HZ03": "direct_model_output",
    "R08HZ04": "derived_physical_indicator",
    "R08HZ05": "derived_physical_indicator",
    "R08HZ06": "derived_physical_indicator",
    "R08HZ07": "derived_physical_indicator",
    "R08HZ08": "derived_physical_indicator",
    "R08HZ09": "derived_physical_indicator",
    "R08HZ11": "deterministic_proxy",
    "R08HZ12": "derived_physical_indicator",
    "R08HZ13": "deterministic_proxy",
    "R09HZ01": "deterministic_proxy",
    "R09HZ02": "deterministic_proxy",
}


@dataclass
class RiskResult:
    asset_daily: pd.DataFrame
    flood_daily: pd.DataFrame
    risk_daily: pd.DataFrame
    risk_summary: pd.DataFrame


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name].astype(float)
    return pd.Series(0.0, index=frame.index)


def _flood_table(
    result: FullModelResult, project: dict[str, Any]
) -> pd.DataFrame:
    components = project["components"]
    sewers = result.component_daily[
        result.component_daily["kind"].eq("sewer")
    ].copy()
    records: list[dict[str, Any]] = []
    for row in sewers.itertuples(index=False):
        component = components[row.component_id]
        overflow_ml = float(getattr(row, "overflow_ml", 0.0))
        depth = max(float(component.get("flood_depth_m", 0.10)), 1e-6)
        width = max(float(component.get("street_flow_width_m", 5.0)), 1e-6)
        shape_factor = float(component.get("flood_shape_factor", 3.0))
        flooded_area = overflow_ml * 1000.0 * shape_factor / depth
        discharge = overflow_ml * 1000.0 / 86400.0
        velocity = discharge / (width * depth)
        velocity_threshold = float(component.get("dangerous_velocity_m_s", 1.0))
        records.append(
            {
                "date": row.date,
                "component_id": row.component_id,
                "overflow_volume_ml": overflow_ml,
                "overflow_volume_m3": overflow_ml * 1000.0,
                "assumed_flood_depth_m": depth,
                "flooded_area_m2": flooded_area,
                "street_velocity_m_s": velocity,
                "high_velocity_flood_area_m2": (
                    flooded_area if velocity >= velocity_threshold else 0.0
                ),
            }
        )
    return pd.DataFrame(records)


def _asset_table(result: FullModelResult) -> pd.DataFrame:
    frame = result.component_daily.copy()
    columns = [
        "date",
        "component_id",
        "kind",
        "expected_failures",
        "expected_outage_hours",
        "maintenance_cost_eur",
        "failure_repair_cost_eur",
        "annualized_capital_cost_eur",
        "capital_cost_eur",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = 0.0
    frame = frame[columns].copy()
    frame.insert(1, "asset_id", frame["component_id"])
    return frame


def evaluate_risks(
    result: FullModelResult,
    project: dict[str, Any],
    timeseries: pd.DataFrame,
) -> RiskResult:
    """Evaluate the 23 official WP33 risk codes with transparent deterministic proxies."""
    system = result.system_daily.reset_index(drop=True)
    areas = result.area_daily
    assets = _asset_table(result)
    floods = _flood_table(result, project)
    flood_by_date = (
        floods.groupby("date", as_index=True)[
            ["flooded_area_m2", "high_velocity_flood_area_m2"]
        ].sum()
        if not floods.empty
        else pd.DataFrame()
    )
    component = result.component_daily
    dates = pd.to_datetime(system["date"])
    thresholds = project.get("risk_thresholds", {})
    target_renewable = float(project.get("risk_settings", {}).get("renewable_target_kwh_day", 0.0))
    baseline_recharge = float(project.get("risk_settings", {}).get("baseline_recharge_ml_day", 0.0))
    baseline_demand = float(project.get("risk_settings", {}).get("baseline_demand_ml_day", system["water_demand_ml"].median()))
    industrial_names = set(project.get("risk_settings", {}).get("industrial_demand_names", ["industrial", "industry"]))

    storage_capacity = sum(
        float(c.get("capacity_ml", 0.0))
        for c in project["components"].values()
        if c["kind"] in {"water_resource", "service_reservoir"}
        and np.isfinite(float(c.get("capacity_ml", 0.0)))
    )
    storage = component[
        component["kind"].isin(["water_resource", "service_reservoir"])
    ].groupby("date")["storage_ml"].sum()
    reservoir_capacity = sum(
        float(c.get("capacity_ml", 0.0))
        for c in project["components"].values()
        if c["kind"] == "service_reservoir"
        and np.isfinite(float(c.get("capacity_ml", 0.0)))
    )
    reservoir_storage = component[
        component["kind"].eq("service_reservoir")
    ].groupby("date")["storage_ml"].sum()
    cso = component.groupby("date")["cso_ml"].sum()
    untreated = component.groupby("date")["untreated_ml"].sum()
    component_costs = component.groupby("date").sum(numeric_only=True)
    area_by_date = areas.groupby("date").sum(numeric_only=True)
    failures = assets.groupby("date").sum(numeric_only=True)

    indicators: dict[str, pd.Series] = {}
    indicators["R01HZ01"] = _series(system, "unmet_demand_ml") * float(
        project.get("risk_settings", {}).get("scarcity_cost_eur_ml", 0.0)
    ) + _series(system, "imported_water_ml") * float(
        project.get("risk_settings", {}).get("import_cost_premium_eur_ml", 0.0)
    )
    indicators["R01HZ02"] = _series(system, "capital_cost_eur")
    indicators["R01HZ03"] = _series(system, "failure_repair_cost_eur") + _series(
        system, "annualized_capital_cost_eur"
    )
    indicators["R01HZ04"] = _series(system, "unmet_demand_ml")
    indicators["R01HZ05"] = _series(system, "electricity_kwh") * float(
        project.get("risk_settings", {}).get("energy_price_stress_eur_kwh", 0.0)
    )
    indicators["R02HZ01"] = (target_renewable - _series(system, "energy_generated_kwh")).clip(lower=0)
    indicators["R03HZ01"] = dates.map(cso).fillna(0.0)
    indicators["R03HZ02"] = dates.map(untreated).fillna(0.0)
    indicators["R05HZ01"] = (
        1.0 - _series(system, "tap_water_quality_index")
    ).clip(lower=0.0)
    indicators["R07HZ01"] = dates.map(
        flood_by_date.get("flooded_area_m2", pd.Series(dtype=float))
    ).fillna(0.0)
    indicators["R07HZ02"] = indicators["R07HZ01"] * float(
        project.get("risk_settings", {}).get("property_flood_fraction", 0.0)
    )
    industrial = pd.Series(0.0, index=system.index)
    for name in industrial_names:
        column = f"unmet_{name}_ml"
        if column in areas:
            industrial = industrial.add(
                dates.map(areas.groupby("date")[column].sum()).fillna(0.0),
                fill_value=0.0,
            )
    indicators["R08HZ03"] = industrial
    indicators["R08HZ04"] = _series(system, "expected_outage_hours") + (
        _series(system, "unmet_demand_ml") > 0
    ).astype(float) * 24.0
    indicators["R08HZ05"] = (
        _series(system, "water_demand_ml") * 1_000_000.0
        / _series(system, "population").replace(0, np.nan)
    ).fillna(0.0)
    indicators["R08HZ06"] = _series(system, "expected_failures")
    if storage_capacity > 0:
        storage_fraction = dates.map(storage).fillna(0.0) / storage_capacity
        indicators["R08HZ07"] = (1.0 - storage_fraction).clip(lower=0.0)
    else:
        indicators["R08HZ07"] = pd.Series(0.0, index=system.index)
    if reservoir_capacity > 0:
        reservoir_fraction = dates.map(reservoir_storage).fillna(0.0) / reservoir_capacity
        indicators["R08HZ08"] = (1.0 - reservoir_fraction).clip(lower=0.0)
    else:
        indicators["R08HZ08"] = pd.Series(0.0, index=system.index)
    recharge = _series(system, "aquifer_recharge_ml")
    indicators["R08HZ09"] = (baseline_recharge - recharge).clip(lower=0.0)
    indicators["R08HZ11"] = _series(system, "unmet_demand_ml")
    indicators["R08HZ12"] = (_series(system, "water_demand_ml") - baseline_demand).clip(lower=0.0)
    indicators["R08HZ13"] = _series(system, "unmet_demand_ml")
    indicators["R09HZ01"] = indicators["R07HZ01"] * float(
        project.get("risk_settings", {}).get("injury_exposure_per_m2", 0.0)
    )
    indicators["R09HZ02"] = dates.map(
        flood_by_date.get("high_velocity_flood_area_m2", pd.Series(dtype=float))
    ).fillna(0.0)

    records: list[dict[str, Any]] = []
    for code, (indicator_name, unit, description) in RISK_CATALOG.items():
        spec = thresholds.get(code, {})
        assessed = "threshold" in spec
        threshold = float(spec["threshold"]) if assessed else np.nan
        tolerance = max(0.0, float(spec.get("tolerance", 1e-9)))
        consequence = float(spec.get("consequence_weight", 1.0))
        values = pd.Series(indicators[code], index=system.index).astype(float)
        if assessed:
            exceedance = (values > threshold + tolerance).astype(float)
            scale = max(
                abs(threshold),
                float(spec.get("normalization_scale", 1.0)),
                tolerance,
                1e-12,
            )
            severity = (
                (values - threshold - tolerance).clip(lower=0.0) / scale
            ).clip(upper=float(spec.get("severity_cap", 1e6)))
        else:
            exceedance = pd.Series(0.0, index=values.index)
            severity = pd.Series(0.0, index=values.index)
        for index, value in values.items():
            records.append(
                {
                    "date": dates.iloc[index],
                    "risk_code": code,
                    "description": description,
                    "indicator": indicator_name,
                    "unit": unit,
                    "value": value,
                    "threshold": threshold,
                    "tolerance": tolerance,
                    "assessed": assessed,
                    "exceeded": bool(exceedance.iloc[index]),
                    "severity": float(severity.iloc[index]),
                    "consequence_weight": consequence,
                    "risk_score": float(exceedance.iloc[index] * severity.iloc[index] * consequence),
                    "method": RISK_METHODS[code],
                }
            )
    daily = pd.DataFrame(records)
    summary = (
        daily.groupby(["risk_code", "description", "indicator", "unit"], as_index=False)
        .agg(
            mean_value=("value", "mean"),
            maximum_value=("value", "max"),
            assessed_days=("assessed", "sum"),
            exceedance_days=("exceeded", "sum"),
            probability=("exceeded", "mean"),
            cumulative_risk_score=("risk_score", "sum"),
            maximum_severity=("severity", "max"),
        )
    )
    summary["assessment_status"] = np.where(
        summary["assessed_days"] > 0, "configured", "unconfigured"
    )
    return RiskResult(assets, floods, daily, summary)
