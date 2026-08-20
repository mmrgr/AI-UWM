from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import date
from math import atan, sqrt
from typing import Any, Mapping

import numpy as np
import pandas as pd


WET_TECHNOLOGIES = {
    "evaporative",
    "efficient_evaporative",
    "liquid_to_water",
    "liquid_water",
}
DRY_TECHNOLOGIES = {"dry", "liquid_to_air", "liquid_air"}
SUPPORTED_COOLING_TECHNOLOGIES = WET_TECHNOLOGIES | DRY_TECHNOLOGIES | {
    "hybrid"
}


def apply_data_center_database(
    component: Mapping[str, Any], database: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Fill omitted research parameters from the auditable AI parameter database."""
    resolved = copy.deepcopy(dict(component))
    if not database:
        return resolved
    cooling = resolved.setdefault("cooling", {})
    technology = cooling.get("technology", "evaporative")
    technology_data = database.get("cooling_technologies", {}).get(technology, {})

    def value(name: str) -> Any:
        record = technology_data.get(name)
        return record.get("value") if isinstance(record, Mapping) else None

    defaults = {
        "cycles_of_concentration": value("typical_coc"),
        "drift_fraction": value("drift_fraction"),
        "heat_to_cooling_fraction": value("heat_to_cooling_fraction"),
    }
    for name, default in defaults.items():
        if default is not None:
            cooling.setdefault(name, default)
    pue = value("typical_pue")
    if pue is not None:
        resolved.setdefault("base_pue", pue)
    penalty = value("technology_pue_adjustment")
    if penalty is not None:
        cooling.setdefault("technology_pue_adjustment", {}).setdefault(technology, penalty)
    grid = database.get("indirect_grid_water_intensity_l_kwh", {})
    if isinstance(grid, Mapping) and grid.get("value") is not None:
        resolved.setdefault("offsite_electricity_water_intensity_l_kwh", grid["value"])
    if cooling.get("coc_mode") == "quality_limited":
        limits = {
            key: record["value"]
            for key, record in database.get("water_quality_thresholds", {}).items()
            if isinstance(record, Mapping) and record.get("value") is not None
        }
        cooling.setdefault("water_quality_limits", limits)
    return resolved


@dataclass(frozen=True)
class DataCenterPlan:
    installed_it_capacity_mw: float
    it_load_mw: float
    load_factor: float
    it_energy_mwh: float
    facility_energy_mwh: float
    pue: float
    cooling_heat_mwh_th: float
    cooling_technology: str
    wet_cooling_fraction: float
    dry_cooling_fraction: float
    wet_bulb_temperature_c: float
    weather_fallback: bool
    cycles_of_concentration: float
    quality_coc_fallback: bool
    evaporation_ml: float
    drift_ml: float
    blowdown_ml: float
    internal_recovery_ml: float
    gross_makeup_ml: float
    external_makeup_ml: float
    consumption_ml: float
    return_flow_ml: float
    wue_l_kwh: float
    target_reclaimed_fraction: float
    offsite_electricity_water_ml: float


def wet_bulb_temperature_c(
    temperature_c: float, relative_humidity_percent: float | None
) -> tuple[float, bool]:
    """Return Stull's wet-bulb approximation and whether dry-bulb fallback was used."""
    if relative_humidity_percent is None or not np.isfinite(relative_humidity_percent):
        return float(temperature_c), True
    rh = float(np.clip(relative_humidity_percent, 0.0, 100.0))
    temperature = float(temperature_c)
    wet_bulb = (
        temperature * atan(0.151977 * sqrt(rh + 8.313659))
        + atan(temperature + rh)
        - atan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * atan(0.023101 * rh)
        - 4.686035
    )
    return float(min(temperature, wet_bulb)), False


def _scheduled_capacity(component: Mapping[str, Any], current_date: date) -> float:
    capacity = float(component.get("installed_it_capacity_mw", 0.0))
    for item in sorted(
        component.get("capacity_schedule", []), key=lambda value: value["date"]
    ):
        if pd.Timestamp(item["date"]).date() <= current_date:
            capacity = float(item["capacity_mw"])
    return max(0.0, capacity)


def _load_factor(
    component: Mapping[str, Any], current_date: date, drivers: Mapping[str, Any]
) -> float:
    load = component.get("load", {})
    mode = component.get("load_mode", load.get("mode", "fixed"))
    base = float(component.get("load_factor", load.get("factor", 0.0)))
    if mode == "timeseries":
        column = component.get("load_factor_column", load.get("timeseries_column"))
        if not column:
            raise ValueError("timeseries load mode requires load_factor_column")
        value = drivers.get(column, drivers.get("load_factor"))
        if value is None:
            raise ValueError(f"timeseries load column is missing: {column}")
        base = float(value)
    elif mode == "profile":
        profile = component.get("load_profile", load.get("profile", {}))
        monthly = profile.get("monthly_factors")
        weekdays = profile.get("weekday_factors")
        if monthly:
            base *= float(monthly[current_date.month - 1])
        if weekdays:
            base *= float(weekdays[current_date.weekday()])
    elif mode != "fixed":
        raise ValueError(f"unsupported data-center load mode: {mode}")
    return float(np.clip(base, 0.0, 1.0))


def _wet_fraction(
    technology: str, wet_bulb_c: float, cooling: Mapping[str, Any]
) -> float:
    if technology in WET_TECHNOLOGIES:
        return float(np.clip(cooling.get("heat_rejection_wet_fraction", 1.0), 0, 1))
    if technology in DRY_TECHNOLOGIES:
        return float(np.clip(cooling.get("heat_rejection_wet_fraction", 0.0), 0, 1))
    if technology != "hybrid":
        raise ValueError(f"unsupported cooling technology: {technology}")
    start = float(cooling.get("hybrid_wet_bulb_start_c", 12.0))
    full = float(cooling.get("hybrid_full_wet_bulb_c", 24.0))
    if full <= start:
        raise ValueError("hybrid_full_wet_bulb_c must exceed hybrid_wet_bulb_start_c")
    return float(np.clip((wet_bulb_c - start) / (full - start), 0.0, 1.0))


def _pue(
    component: Mapping[str, Any], temperature_c: float, humidity: float | None,
    load_factor: float, technology: str,
) -> float:
    base = float(component.get("base_pue", component.get("pue", 1.2)))
    mode = component.get("pue_mode", "fixed")
    default_adjustments = {
        "efficient_evaporative": -0.02,
        "hybrid": 0.04,
        "dry": 0.12,
        "liquid_to_air": 0.05,
        "liquid_air": 0.05,
        "liquid_to_water": -0.02,
        "liquid_water": -0.02,
    }
    cooling = component.get("cooling", {})
    adjustments = cooling.get("technology_pue_adjustment", {})
    technology_adjustment = float(
        adjustments.get(technology, default_adjustments.get(technology, 0.0))
    )
    if mode in {"fixed", "fixed_pue"}:
        return float(np.clip(base + technology_adjustment, 1.0, 2.5))
    if mode not in {"dynamic", "dynamic_pue"}:
        raise ValueError(f"unsupported PUE mode: {mode}")
    value = base
    value += float(component.get("temperature_coefficient", 0.0)) * (
        temperature_c - float(component.get("reference_temperature_c", 20.0))
    )
    if humidity is not None and np.isfinite(humidity):
        value += float(component.get("humidity_coefficient", 0.0)) * (
            float(humidity) - float(component.get("reference_relative_humidity", 50.0))
        )
    value += float(component.get("load_coefficient", 0.0)) * (
        load_factor - float(component.get("reference_load_factor", 1.0))
    )
    value += technology_adjustment
    return float(
        np.clip(
            value,
            float(component.get("min_pue", 1.0)),
            float(component.get("max_pue", 2.5)),
        )
    )


def _cycles_of_concentration(
    component: Mapping[str, Any], cooling: Mapping[str, Any]
) -> tuple[float, bool]:
    default_coc = 7.0 if cooling.get("technology") == "efficient_evaporative" else 5.0
    design = max(1.000001, float(cooling.get("cycles_of_concentration", default_coc)))
    if cooling.get("coc_mode", "fixed") != "quality_limited":
        return design, False
    limits = cooling.get("water_quality_limits", {})
    sources = component.get("water_sources", {})
    concentrations: dict[str, float] = {}
    for source in sources.values():
        fraction = float(source.get("target_fraction", 0.0))
        for indicator, value in source.get("quality", {}).items():
            concentrations[indicator] = concentrations.get(indicator, 0.0) + fraction * float(value)
    ratios = [
        float(limit) / concentrations[indicator]
        for indicator, limit in limits.items()
        if concentrations.get(indicator, 0.0) > 0
    ]
    if not ratios:
        return design, True
    return max(1.000001, min(design, min(ratios))), False


def calculate_data_center_plan(
    component: Mapping[str, Any],
    current_date: pd.Timestamp,
    drivers: Mapping[str, Any],
) -> DataCenterPlan:
    temperature = float(drivers.get("temperature_c", 20.0))
    humidity_value = drivers.get("relative_humidity", drivers.get("relative_humidity_pct"))
    humidity = None if humidity_value is None else float(humidity_value)
    wet_bulb, weather_fallback = wet_bulb_temperature_c(temperature, humidity)
    capacity = _scheduled_capacity(component, current_date.date())
    load_factor = _load_factor(component, current_date.date(), drivers)
    it_load = capacity * load_factor
    it_energy = it_load * 24.0
    cooling = component.get("cooling", {})
    technology = str(cooling.get("technology", "evaporative"))
    wet_fraction = _wet_fraction(technology, wet_bulb, cooling)
    pue = _pue(component, temperature, humidity, load_factor, technology)
    facility_energy = it_energy * pue
    heat = it_energy * float(cooling.get("heat_to_cooling_fraction", 1.0))
    heat *= max(
        0.0,
        1.0
        + float(cooling.get("heat_rejection_temperature_coefficient", 0.0))
        * (temperature - float(cooling.get("reference_temperature_c", 20.0))),
    )
    wet_heat = heat * wet_fraction
    latent_heat = max(1e-12, float(cooling.get("latent_heat_kj_kg", 2450.0)))
    evaporation = wet_heat * 3_600_000.0 / latent_heat / 1_000_000.0
    coc, quality_fallback = _cycles_of_concentration(component, cooling)
    gross_without_drift_detail = evaporation * coc / (coc - 1.0) if evaporation else 0.0
    default_drift = 0.00005 if technology == "efficient_evaporative" else 0.0002
    drift = gross_without_drift_detail * float(
        cooling.get(
            "drift_fraction_of_makeup",
            cooling.get("drift_fraction", default_drift),
        )
    )
    blowdown = max(evaporation / (coc - 1.0) - drift, 0.0) if evaporation else 0.0
    gross = evaporation + drift + blowdown
    internal_recovery = blowdown * float(cooling.get("internal_recovery_fraction", 0.0))
    returnable_blowdown = max(0.0, blowdown - internal_recovery)
    return_flow = returnable_blowdown * float(
        cooling.get("blowdown_return_fraction", 1.0)
    )
    consumption = evaporation + drift + returnable_blowdown - return_flow
    external = consumption + return_flow
    wue = external * 1000.0 / it_energy if it_energy else 0.0
    reclaimed_target = float(
        component.get("water_sources", {}).get("reclaimed", {}).get("target_fraction", 0.0)
    )
    offsite_intensity = float(
        component.get("offsite_electricity_water_intensity_l_kwh", 0.0)
    )
    return DataCenterPlan(
        installed_it_capacity_mw=capacity,
        it_load_mw=it_load,
        load_factor=load_factor,
        it_energy_mwh=it_energy,
        facility_energy_mwh=facility_energy,
        pue=pue,
        cooling_heat_mwh_th=heat,
        cooling_technology=technology,
        wet_cooling_fraction=wet_fraction,
        dry_cooling_fraction=1.0 - wet_fraction,
        wet_bulb_temperature_c=wet_bulb,
        weather_fallback=weather_fallback,
        cycles_of_concentration=coc,
        quality_coc_fallback=quality_fallback,
        evaporation_ml=evaporation,
        drift_ml=drift,
        blowdown_ml=blowdown,
        internal_recovery_ml=internal_recovery,
        gross_makeup_ml=gross,
        external_makeup_ml=external,
        consumption_ml=consumption,
        return_flow_ml=return_flow,
        wue_l_kwh=wue,
        target_reclaimed_fraction=reclaimed_target,
        offsite_electricity_water_ml=facility_energy * offsite_intensity / 1000.0,
    )


def finalize_data_center_day(
    plan: DataCenterPlan,
    *,
    external_withdrawal_ml: float,
    reclaimed_water_ml: float,
    potable_water_ml: float,
    other_water_ml: float,
    storage_start_ml: float,
    storage_capacity_ml: float,
) -> dict[str, Any]:
    withdrawal = max(0.0, float(external_withdrawal_ml))
    available = storage_start_ml + withdrawal
    process_external = min(plan.external_makeup_ml, available)
    scale = process_external / plan.external_makeup_ml if plan.external_makeup_ml else 1.0
    storage_end = min(
        max(0.0, storage_capacity_ml), max(0.0, available - process_external)
    )
    values = asdict(plan)
    for name in (
        "evaporation_ml",
        "drift_ml",
        "blowdown_ml",
        "internal_recovery_ml",
        "gross_makeup_ml",
        "external_makeup_ml",
        "consumption_ml",
        "return_flow_ml",
    ):
        values[name] *= scale
    values.update(
        {
            "external_withdrawal_ml": withdrawal,
            "potable_water_ml": max(0.0, potable_water_ml),
            "reclaimed_water_ml": max(0.0, reclaimed_water_ml),
            "other_water_ml": max(0.0, other_water_ml),
            "storage_start_ml": storage_start_ml,
            "storage_end_ml": storage_end,
            "storage_change_ml": storage_end - storage_start_ml,
            "unmet_cooling_water_ml": max(
                0.0, plan.external_makeup_ml - process_external
            ),
            "actual_reclaimed_fraction": (
                max(0.0, reclaimed_water_ml) / withdrawal if withdrawal else 0.0
            ),
            "direct_urban_withdrawal_ml": withdrawal,
            "direct_urban_consumption_ml": values["consumption_ml"],
            "total_water_footprint_ml": (
                values["consumption_ml"] + plan.offsite_electricity_water_ml
            ),
        }
    )
    values["water_balance_residual_ml"] = (
        withdrawal
        - values["consumption_ml"]
        - values["return_flow_ml"]
        - values["storage_change_ml"]
    )
    return values
