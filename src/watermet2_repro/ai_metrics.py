from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .full_engine import FullModelResult


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _max_component_utilization(
    result: FullModelResult,
    project: dict[str, Any] | None,
    kind: str,
    flow_column: str,
    capacity_keys: tuple[str, ...],
) -> float:
    if not project:
        return 0.0
    maximum = 0.0
    for component_id, component in project.get("components", {}).items():
        if component.get("kind") != kind:
            continue
        capacity = next(
            (float(component[key]) for key in capacity_keys if component.get(key) is not None),
            0.0,
        )
        if capacity <= 0:
            continue
        rows = result.component_daily[result.component_daily["component_id"] == component_id]
        if not rows.empty and flow_column in rows:
            maximum = max(maximum, float((rows[flow_column] / capacity).max()))
    return maximum


def summarize_ai_water_kpis(
    result: FullModelResult, project: dict[str, Any] | None = None
) -> dict[str, float]:
    """Return paper-ready AI water, peak, circularity and infrastructure KPIs."""
    dc = result.data_center_daily.copy()
    if dc.empty:
        dc = pd.DataFrame(
            {name: np.zeros(len(result.system_daily)) for name in (
                "external_withdrawal_ml", "potable_water_ml", "reclaimed_water_ml",
                "consumption_ml", "return_flow_ml", "unmet_cooling_water_ml",
                "it_energy_mwh", "facility_energy_mwh", "offsite_electricity_water_ml",
            )},
            index=result.system_daily.index,
        )
        dc["date"] = result.system_daily["date"].to_numpy()
    daily = dc.groupby("date", as_index=False).sum(numeric_only=True)
    withdrawal = float(daily["external_withdrawal_ml"].sum())
    potable = float(daily["potable_water_ml"].sum())
    reclaimed = float(daily["reclaimed_water_ml"].sum())
    consumption = float(daily["consumption_ml"].sum())
    return_flow = float(daily["return_flow_ml"].sum())
    unmet_ai = float(daily["unmet_cooling_water_ml"].sum())
    rolling_7 = daily["external_withdrawal_ml"].rolling(7, min_periods=1).mean()
    summer = daily[pd.to_datetime(daily["date"]).dt.month.isin((6, 7, 8))]
    system = result.system_daily
    demand = float(system["water_demand_ml"].sum())
    delivered = float(system["delivered_total_ml"].sum())
    domestic_unmet = max(0.0, float(system["unmet_demand_ml"].sum()) - unmet_ai)
    if domestic_unmet < 1e-9:
        domestic_unmet = 0.0
    supply_capacity = 0.0
    if project:
        supply_capacity = sum(
            float(component.get("daily_capacity_ml", component.get("capacity_ml", 0.0)))
            for component in project.get("components", {}).values()
            if component.get("kind") == "wtw"
        )
    max_system_demand = float(system["water_demand_ml"].max())
    dc_peak_date = pd.Timestamp(daily.loc[daily["external_withdrawal_ml"].idxmax(), "date"])
    city_peak_date = pd.Timestamp(system.loc[system["water_demand_ml"].idxmax(), "date"])
    values = {
        "total_withdrawal_ml": withdrawal,
        "freshwater_withdrawal_ml": potable,
        "reclaimed_water_use_ml": reclaimed,
        "consumption_ml": consumption,
        "return_flow_ml": return_flow,
        "maximum_daily_withdrawal_ml": float(daily["external_withdrawal_ml"].max()),
        "p95_daily_withdrawal_ml": float(daily["external_withdrawal_ml"].quantile(0.95)),
        "maximum_7day_average_ml": float(rolling_7.max()),
        "summer_peak_withdrawal_ml": float(summer["external_withdrawal_ml"].max()) if not summer.empty else 0.0,
        "unmet_cooling_water_ml": unmet_ai,
        "freshwater_dependency_ratio": _safe_ratio(potable, withdrawal + unmet_ai),
        "reclaimed_water_substitution_ratio": _safe_ratio(reclaimed, withdrawal + unmet_ai),
        "urban_water_circularity_ratio": _safe_ratio(reclaimed, withdrawal),
        "consumption_fraction": _safe_ratio(consumption, withdrawal),
        "return_ratio": _safe_ratio(return_flow, withdrawal),
        "system_reliability_fraction": _safe_ratio(delivered, demand) if demand else 1.0,
        "domestic_unmet_ml": domestic_unmet,
        "peak_capacity_ratio": _safe_ratio(max_system_demand, supply_capacity),
        "coincident_city_ai_peak": float(abs((city_peak_date - dc_peak_date).days) <= 7),
        "max_wtw_utilization": _max_component_utilization(result, project, "wtw", "outflow_ml", ("daily_capacity_ml", "capacity_ml")),
        "max_wwtw_utilization": _max_component_utilization(result, project, "wwtw", "inflow_ml", ("daily_capacity_ml", "capacity_ml")),
        "max_reuse_utilization": _max_component_utilization(result, project, "reuse", "outflow_ml", ("treatment_capacity_ml_day", "capacity_ml")),
        "wastewater_ml": float(result.component_daily.loc[result.component_daily["kind"] == "wwtw", "inflow_ml"].sum()),
        "system_energy_kwh": float(system["electricity_kwh"].sum()),
        "system_carbon_kg_co2e": float(system["ghg_net_kg_co2e"].sum()),
        "it_energy_mwh": float(daily["it_energy_mwh"].sum()),
        "facility_energy_mwh": float(daily["facility_energy_mwh"].sum()),
        "offsite_electricity_water_ml": float(daily["offsite_electricity_water_ml"].sum()),
    }
    return values


def compare_baseline_ai(
    baseline: FullModelResult,
    ai_scenario: FullModelResult,
    baseline_project: dict[str, Any] | None = None,
    ai_project: dict[str, Any] | None = None,
) -> pd.DataFrame:
    baseline_values = summarize_ai_water_kpis(baseline, baseline_project)
    ai_values = summarize_ai_water_kpis(ai_scenario, ai_project)
    keys = sorted(set(baseline_values) | set(ai_values))
    return pd.DataFrame(
        {
            "metric": keys,
            "baseline": [baseline_values.get(key, 0.0) for key in keys],
            "ai_scenario": [ai_values.get(key, 0.0) for key in keys],
            "delta": [ai_values.get(key, 0.0) - baseline_values.get(key, 0.0) for key in keys],
        }
    )


def infrastructure_utilization_summary(
    result: FullModelResult, project: dict[str, Any]
) -> pd.DataFrame:
    """Return daily, monthly and annual maxima for WTW, WWTW and reuse assets."""
    definitions = {
        "wtw": ("outflow_ml", ("daily_capacity_ml", "capacity_ml")),
        "wwtw": ("inflow_ml", ("daily_capacity_ml", "capacity_ml")),
        "reuse": ("outflow_ml", ("treatment_capacity_ml_day", "capacity_ml")),
    }
    records: list[dict[str, Any]] = []
    for component_id, component in project.get("components", {}).items():
        kind = component.get("kind")
        if kind not in definitions:
            continue
        flow_column, capacity_keys = definitions[kind]
        capacity = next((float(component[key]) for key in capacity_keys if component.get(key) is not None), 0.0)
        rows = result.component_daily[result.component_daily.component_id == component_id].copy()
        if capacity <= 0 or rows.empty:
            continue
        rows["utilization"] = rows[flow_column] / capacity
        rows["date"] = pd.to_datetime(rows["date"])
        periods = {
            "daily": rows.assign(period=rows["date"].dt.strftime("%Y-%m-%d")),
            "monthly": rows.assign(period=rows["date"].dt.strftime("%Y-%m")),
            "annual": rows.assign(period=rows["date"].dt.strftime("%Y")),
        }
        for scale, frame in periods.items():
            for period, maximum in frame.groupby("period")["utilization"].max().items():
                records.append({
                    "scale": scale,
                    "period": period,
                    "component_id": component_id,
                    "kind": kind,
                    "maximum_utilization": float(maximum),
                })
    return pd.DataFrame(records)
