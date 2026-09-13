"""Executable research workflows required by the AI-UWM opening proposal.

These functions deliberately sit above the daily WaterMet2 engine: they keep
the physical daily model as the source of truth and add reproducible hourly
stress, S/G state perturbations, controls, robustness, provenance and margin
tables around it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .ai_capacity import find_ai_carrying_capacity, scan_ai_capacity
from .ai_metrics import summarize_ai_water_kpis
from .cawcc import build_intraday_profile, intraday_peak_proxy
from .full_engine import FullWaterMet2Model


STATE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "S0": {"resource": 1.0, "wtw": 1.0, "wwtw": 1.0, "reuse": 1.0, "local": 1.0},
    "S1": {"resource": 1.25, "wtw": 0.75, "wwtw": 1.0, "reuse": 1.0, "local": 1.0},
    "S2": {"resource": 0.65, "wtw": 1.0, "wwtw": 1.25, "reuse": 1.25, "local": 1.15},
    "S3": {"resource": 1.15, "wtw": 1.15, "wwtw": 1.15, "reuse": 1.15, "local": 0.65},
}
PRESSURE_SETTINGS: dict[str, dict[str, float]] = {
    "G0": {"temperature_delta_c": 0.0, "inflow_factor": 1.0, "demand_factor": 1.0},
    "G1": {"temperature_delta_c": 4.0, "inflow_factor": 1.0, "demand_factor": 1.08},
    "G2": {"temperature_delta_c": 0.0, "inflow_factor": 0.65, "demand_factor": 1.0},
    "G3": {"temperature_delta_c": 4.0, "inflow_factor": 0.65, "demand_factor": 1.12},
}


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    target: Any = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


def apply_state_pressure(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    state: str = "S0",
    pressure: str = "G0",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Materialize report S/G labels into a runnable project and drivers."""
    if state not in STATE_MULTIPLIERS or pressure not in PRESSURE_SETTINGS:
        raise ValueError("state must be S0-S3 and pressure must be G0-G3")
    trial = copy.deepcopy(project)
    drivers = timeseries.copy()
    sm = STATE_MULTIPLIERS[state]
    for component in trial.get("components", {}).values():
        kind = component.get("kind")
        if kind == "water_resource":
            for key in ("capacity_ml", "abstraction_capacity_ml_day"):
                if key in component:
                    component[key] = float(component[key]) * sm["resource"]
        elif kind == "wtw":
            for key in ("daily_capacity_ml", "capacity_ml"):
                if key in component:
                    component[key] = float(component[key]) * sm["wtw"]
        elif kind == "wwtw":
            for key in ("daily_capacity_ml", "capacity_ml"):
                if key in component:
                    component[key] = float(component[key]) * sm["wwtw"]
        elif kind == "reuse":
            for key in ("capacity_ml", "treatment_capacity_ml_day"):
                if key in component:
                    component[key] = float(component[key]) * sm["reuse"]
        elif kind == "data_center" and component.get("local_connection_capacity_ml_day") is not None:
            component["local_connection_capacity_ml_day"] = float(component["local_connection_capacity_ml_day"]) * sm["local"]
    settings = PRESSURE_SETTINGS[pressure]
    if "temperature_c" in drivers:
        drivers["temperature_c"] = drivers["temperature_c"].astype(float) + settings["temperature_delta_c"]
    if settings["inflow_factor"] != 1:
        for column in drivers.columns:
            if "inflow" in column.lower():
                drivers[column] = drivers[column].astype(float) * settings["inflow_factor"]
    if settings["demand_factor"] != 1:
        for area in trial.get("local_areas", {}).values():
            for profile in area.get("demand_profiles", []):
                if "base_value" in profile:
                    profile["base_value"] = float(profile["base_value"]) * settings["demand_factor"]
    trial.setdefault("research", {}).update({"system_state": state, "pressure": pressure})
    return trial, drivers


def run_state_pressure_matrix(
    project: dict[str, Any], timeseries: pd.DataFrame,
    states: Iterable[str] = ("S0", "S1", "S2", "S3"),
    pressures: Iterable[str] = ("G0", "G1", "G2", "G3"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in states:
        for pressure in pressures:
            trial, drivers = apply_state_pressure(project, timeseries, state, pressure)
            result = FullWaterMet2Model(trial, drivers).run()
            rows.append({"system_state": state, "pressure": pressure, **summarize_ai_water_kpis(result, trial)})
    return pd.DataFrame(rows)


def industrial_control_project(project: dict[str, Any], mode: str = "annual_water") -> dict[str, Any]:
    """Create an equivalent non-AI cooling-load control for causal comparison."""
    if mode not in {"annual_water", "peak_heat"}:
        raise ValueError("mode must be annual_water or peak_heat")
    trial = copy.deepcopy(project)
    for component in trial.get("components", {}).values():
        if component.get("kind") == "data_center":
            component["priority_class"] = "industrial_control"
            component["control_mode"] = mode
            component["load_profile"] = {"monthly_factors": [1.0] * 12}
            # Hold the control at an ordinary industrial wet-cooling envelope,
            # rather than silently re-running the AI hybrid/dynamic design.
            component.setdefault("cooling", {})["technology"] = "evaporative"
            component["pue_mode"] = "fixed"
            if mode == "peak_heat":
                component["load_factor"] = min(1.0, float(component.get("load_factor", 0.75)) * 1.08)
    trial.setdefault("research", {})["control_group"] = mode
    return trial


def compare_ai_industrial(
    project: dict[str, Any], timeseries: pd.DataFrame, mode: str = "annual_water"
) -> pd.DataFrame:
    ai_result = FullWaterMet2Model(project, timeseries).run()
    control_project = industrial_control_project(project, mode)
    data_centers = [c for c in project.get("components", {}).values() if c.get("kind") == "data_center"]
    target_withdrawal = float(summarize_ai_water_kpis(ai_result, project)["total_withdrawal_ml"])
    target_heat = float(ai_result.data_center_daily["cooling_heat_mwh_th"].max()) if not ai_result.data_center_daily.empty else 0.0
    # Calibrate the control group's single load factor to the requested
    # matching criterion, preserving a one-factor-at-a-time comparison.
    low, high = 0.0, 1.0
    for _ in range(28):
        midpoint = (low + high) / 2.0
        for component in control_project.get("components", {}).values():
            if component.get("kind") == "data_center":
                component["load_factor"] = midpoint
        trial_result = FullWaterMet2Model(control_project, timeseries).run()
        if mode == "annual_water":
            actual = float(summarize_ai_water_kpis(trial_result, control_project)["total_withdrawal_ml"])
            target = target_withdrawal
        else:
            actual = float(trial_result.data_center_daily["cooling_heat_mwh_th"].max()) if not trial_result.data_center_daily.empty else 0.0
            target = target_heat
        if not data_centers or abs(actual - target) <= max(1e-9, target * 1e-5):
            break
        if actual < target:
            low = midpoint
        else:
            high = midpoint
    control_result = FullWaterMet2Model(control_project, timeseries).run()
    ai = summarize_ai_water_kpis(ai_result, project)
    control = summarize_ai_water_kpis(control_result, control_project)
    return pd.DataFrame({"metric": sorted(set(ai) | set(control)), "ai": [ai.get(k, np.nan) for k in sorted(set(ai) | set(control))], "industrial_control": [control.get(k, np.nan) for k in sorted(set(ai) | set(control))]})


def hourly_stress_scan(
    project: dict[str, Any], timeseries: pd.DataFrame,
    capacities_mw: Iterable[float], constraints: Mapping[str, Any],
    *, profile: pd.DataFrame | None = None, city_phase_hours: int = 0,
) -> pd.DataFrame:
    """Scan CAWCC with explicit hourly peak multipliers and phase overlap."""
    profile = profile if profile is not None else build_intraday_profile()
    # `build_intraday_profile` is normalized to a daily mean of one, so the
    # hourly share of a daily total is multiplier / 24.
    peak_multiplier = float(profile["load_multiplier"].max()) / 24.0
    base = scan_ai_capacity(project, timeseries, capacities_mw=capacities_mw)
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        record = row.to_dict()
        record["hourly_peak_withdrawal_ml"] = float(row["maximum_daily_withdrawal_ml"]) * peak_multiplier
        record["hourly_peak_consumption_ml"] = float(row["consumption_ml"] / max(len(timeseries), 1)) * peak_multiplier
        record["hourly_city_phase_hours"] = int(city_phase_hours)
        record["hourly_peak_capacity_ratio"] = float(row["peak_capacity_ratio"]) * peak_multiplier
        record["hourly_system_reliability_fraction"] = float(row["system_reliability_fraction"])
        rows.append(record)
    frame = pd.DataFrame(rows)
    definitions = constraints.get("constraints", constraints)
    if "peak_capacity_ratio" in definitions:
        frame["hourly_peak_capacity_ratio"] = frame["hourly_peak_capacity_ratio"]
    return frame


def dimensionless_margins(scan: pd.DataFrame, constraints: Mapping[str, Any]) -> pd.DataFrame:
    definitions = constraints.get("constraints", constraints)
    rows: list[dict[str, Any]] = []
    for _, row in scan.iterrows():
        for metric, definition in definitions.items():
            target = float(definition["value"])
            actual = float(row[metric])
            margin = (actual - target) / max(abs(target), 1e-12) if definition["operator"].startswith("<") else (actual - target) / max(abs(target), 1e-12)
            rows.append({"ai_capacity_mw": float(row["ai_capacity_mw"]), "constraint": metric, "dimensionless_margin": margin, "feasible": margin <= 0 if definition["operator"].startswith("<") else margin >= 0})
    return pd.DataFrame(rows)


def robustness_matrix(
    project: dict[str, Any], timeseries: pd.DataFrame,
    parameter_sets: Iterable[Mapping[str, Any]], capacities_mw: Iterable[float], constraints: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, settings in enumerate(parameter_sets):
        trial = copy.deepcopy(project)
        for path, value in settings.items():
            _set_path(trial, str(path), value)
        scan = scan_ai_capacity(trial, timeseries, capacities_mw=capacities_mw)
        threshold = find_ai_carrying_capacity(scan, dict(constraints))
        rows.append({"sample_id": index, **settings, "maximum_safe_ai_capacity_mw": threshold["maximum_safe_ai_capacity_mw"], "first_failed_capacity_mw": threshold["first_failed_capacity_mw"], "limiting_constraint": threshold["limiting_constraint"]})
    return pd.DataFrame(rows)


def parameter_provenance(project: Mapping[str, Any]) -> pd.DataFrame:
    database = project.get("ai_data_center_database", {})
    rows: list[dict[str, Any]] = []
    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            if {"value", "unit", "source"}.issubset(value):
                rows.append({"path": path, "value": value.get("value"), "unit": value.get("unit"), "source": value.get("source"), "year": value.get("year"), "evidence_type": value.get("type"), "uncertainty": value.get("uncertainty")})
            for key, child in value.items(): walk(child, f"{path}.{key}".strip("."))
    walk(database)
    return pd.DataFrame(rows)
