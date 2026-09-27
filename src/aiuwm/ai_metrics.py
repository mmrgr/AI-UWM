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


def _max_source_abstraction_ratio(
    result: FullModelResult, project: dict[str, Any] | None
) -> float:
    """Maximum daily abstraction / abstraction-capacity ratio over all sources.

    The engine caps abstraction at `abstraction_capacity_ml_day`, so this ratio
    can never exceed one; it is meant to be used with a planning-margin
    threshold (for example 0.95) rather than with `<= 1`.
    """
    if not project:
        return 0.0
    maximum = 0.0
    for component_id, component in project.get("components", {}).items():
        if component.get("kind") != "water_resource":
            continue
        capacity = float(component.get("abstraction_capacity_ml_day", 0.0) or 0.0)
        if capacity <= 0:
            continue
        rows = result.component_daily[result.component_daily["component_id"] == component_id]
        if not rows.empty and "outflow_ml" in rows:
            maximum = max(maximum, float((rows["outflow_ml"] / capacity).max()))
    return maximum


def _mean_source_abstraction_ratio(
    result: FullModelResult, project: dict[str, Any] | None
) -> float:
    if not project:
        return 0.0
    maximum = 0.0
    for component_id, component in project.get("components", {}).items():
        if component.get("kind") != "water_resource":
            continue
        capacity = float(component.get("abstraction_capacity_ml_day", 0.0) or 0.0)
        if capacity <= 0:
            continue
        rows = result.component_daily[result.component_daily["component_id"] == component_id]
        if not rows.empty and "outflow_ml" in rows:
            maximum = max(maximum, float(rows["outflow_ml"].mean()) / capacity)
    return maximum


def _min_source_storage_ml(
    result: FullModelResult, project: dict[str, Any] | None
) -> float:
    """Minimum end-of-day storage across all water sources (ML)."""
    if not project:
        return float("nan")
    minimum = float("nan")
    for component_id, component in project.get("components", {}).items():
        if component.get("kind") != "water_resource":
            continue
        rows = result.component_daily[result.component_daily["component_id"] == component_id]
        if rows.empty or "storage_ml" not in rows:
            continue
        value = float(rows["storage_ml"].min())
        minimum = value if np.isnan(minimum) else min(minimum, value)
    return minimum


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
                "internal_recovery_ml", "gross_makeup_ml",
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
    process_column = "external_makeup_ml" if "external_makeup_ml" in daily else "gross_makeup_ml"
    process_makeup = float(daily[process_column].sum())
    rolling_7 = daily["external_withdrawal_ml"].rolling(7, min_periods=1).mean()
    summer = daily[pd.to_datetime(daily["date"]).dt.month.isin((6, 7, 8))]
    system = result.system_daily
    demand = float(system["water_demand_ml"].sum())
    delivered = float(system["delivered_total_ml"].sum())
    # Household shortage is recorded directly by the allocation engine.  A
    # system-minus-AI residual also includes industrial and municipal shortage.
    domestic_unmet = float(result.area_daily["unmet_domestic_ml"].sum()) if "unmet_domestic_ml" in result.area_daily else 0.0
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
    local_connection_ratios: list[float] = []
    mean_local_connection_ratios: list[float] = []
    grid_connection_ratios: list[float] = []
    mean_grid_connection_ratios: list[float] = []
    if project and not result.data_center_daily.empty:
        for component_id, component in project.get("components", {}).items():
            if component.get("kind") != "data_center":
                continue
            rows = result.data_center_daily[
                result.data_center_daily["data_center_id"] == component_id
            ]
            if rows.empty:
                continue
            local_capacity = component.get("local_connection_capacity_ml_day")
            if local_capacity is not None and float(local_capacity) > 0:
                area_rows = result.area_daily[
                    result.area_daily["area_id"] == component.get("local_area")
                ]
                demand_column = f"demand_data_center::{component_id}_ml"
                if demand_column in area_rows:
                    local_connection_ratios.append(
                        float(area_rows[demand_column].max()) / float(local_capacity)
                    )
                    mean_local_connection_ratios.append(
                        float(area_rows[demand_column].mean()) / float(local_capacity)
                    )
            grid_capacity = component.get("grid_connection_capacity_mw")
            if grid_capacity is not None and float(grid_capacity) > 0:
                grid_connection_ratios.append(
                    float((rows["facility_energy_mwh"] / 24.0).max())
                    / float(grid_capacity)
                )
                mean_grid_connection_ratios.append(
                    float(rows["facility_energy_mwh"].mean()) / 24.0 / float(grid_capacity)
                )
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
        # This is a total-city-demand / potable-WTW-capacity ratio.  Keep the
        # historical key for reproducibility, but expose the denominator in a
        # descriptive key so it is not confused with total system capacity.
        "peak_total_demand_to_wtw_capacity": _safe_ratio(max_system_demand, supply_capacity),
        "peak_capacity_ratio": _safe_ratio(max_system_demand, supply_capacity),
        "mean_total_demand_to_wtw_capacity": _safe_ratio(float(system["water_demand_ml"].mean()), supply_capacity),
        "coincident_city_ai_peak": float(abs((city_peak_date - dc_peak_date).days) <= 7),
        "max_wtw_utilization": _max_component_utilization(result, project, "wtw", "outflow_ml", ("daily_capacity_ml", "capacity_ml")),
        "max_wwtw_utilization": _max_component_utilization(result, project, "wwtw", "inflow_ml", ("daily_capacity_ml", "capacity_ml")),
        "max_reuse_utilization": _max_component_utilization(result, project, "reuse", "outflow_ml", ("treatment_capacity_ml_day", "capacity_ml")),
        "max_source_abstraction_ratio": _max_source_abstraction_ratio(result, project),
        "mean_source_abstraction_ratio": _mean_source_abstraction_ratio(result, project),
        "min_source_storage_ml": _min_source_storage_ml(result, project),
        "wastewater_ml": float(result.component_daily.loc[result.component_daily["kind"] == "wwtw", "inflow_ml"].sum()),
        "system_energy_kwh": float(system["electricity_kwh"].sum()),
        "system_carbon_kg_co2e": float(system["ghg_net_kg_co2e"].sum()),
        "it_energy_mwh": float(daily["it_energy_mwh"].sum()),
        "facility_energy_mwh": float(daily["facility_energy_mwh"].sum()),
        "site_wue_l_kwh_it": _safe_ratio(withdrawal * 1000.0, float(daily["it_energy_mwh"].sum())),
        "process_wue_l_kwh_it": _safe_ratio(process_makeup * 1000.0, float(daily["it_energy_mwh"].sum())),
        "wue_l_kwh_it": _safe_ratio(withdrawal * 1000.0, float(daily["it_energy_mwh"].sum())),
        "offsite_electricity_water_ml": float(daily["offsite_electricity_water_ml"].sum()),
        "max_local_connection_ratio": max(local_connection_ratios) if local_connection_ratios else float("nan"),
        # These are annual/period means in time, while the scalar retains the
        # most constrained component when several data centres are present.
        # That matches the daily max metrics used for the infrastructure gate.
        "mean_local_connection_ratio": max(mean_local_connection_ratios) if mean_local_connection_ratios else float("nan"),
        "max_daily_average_grid_connection_ratio": max(grid_connection_ratios) if grid_connection_ratios else float("nan"),
        "mean_daily_average_grid_connection_ratio": max(mean_grid_connection_ratios) if mean_grid_connection_ratios else float("nan"),
        "internal_cooling_recovery_ratio": _safe_ratio(
            float(daily["internal_recovery_ml"].sum()),
            float(daily["gross_makeup_ml"].sum()),
        ),
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
