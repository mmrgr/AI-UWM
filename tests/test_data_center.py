from __future__ import annotations

import numpy as np
import pandas as pd

from watermet2_repro.data_center import (
    calculate_data_center_plan,
    finalize_data_center_day,
    wet_bulb_temperature_c,
)


def component(technology: str = "evaporative", capacity_mw: float = 500.0):
    return {
        "kind": "data_center",
        "local_area": "LA1",
        "installed_it_capacity_mw": capacity_mw,
        "load_mode": "fixed",
        "load_factor": 0.75,
        "pue_mode": "dynamic",
        "base_pue": 1.20,
        "temperature_coefficient": 0.003,
        "reference_temperature_c": 20.0,
        "load_coefficient": -0.05,
        "min_pue": 1.05,
        "max_pue": 1.60,
        "cooling": {
            "technology": technology,
            "heat_to_cooling_fraction": 1.0,
            "latent_heat_kj_kg": 2450.0,
            "cycles_of_concentration": 5.0,
            "drift_fraction_of_makeup": 0.0002,
            "blowdown_return_fraction": 1.0,
            "internal_recovery_fraction": 0.0,
            "hybrid_wet_bulb_start_c": 12.0,
            "hybrid_full_wet_bulb_c": 24.0,
            "technology_pue_adjustment": {
                "dry": 0.12,
                "liquid_to_air": 0.06,
                "liquid_to_water": -0.02,
            },
        },
        "water_sources": {
            "reclaimed": {"target_fraction": 0.7, "priority": 1},
            "potable": {"target_fraction": 0.3, "priority": 2},
        },
        "water_fallback": True,
        "offsite_electricity_water_intensity_l_kwh": 1.2,
    }


def test_stull_wet_bulb_is_physical_and_missing_rh_falls_back() -> None:
    wet_bulb, fallback = wet_bulb_temperature_c(30.0, 50.0)
    assert 20.0 < wet_bulb < 25.0
    assert not fallback
    dry_bulb, fallback = wet_bulb_temperature_c(30.0, None)
    assert dry_bulb == 30.0
    assert fallback


def test_capacity_schedule_energy_and_external_water_balance() -> None:
    config = component()
    config["capacity_schedule"] = [
        {"date": "2027-01-01", "capacity_mw": 100.0},
        {"date": "2028-01-01", "capacity_mw": 250.0},
    ]
    plan = calculate_data_center_plan(
        config,
        pd.Timestamp("2028-06-01"),
        {"temperature_c": 30.0, "relative_humidity": 50.0},
    )
    assert plan.installed_it_capacity_mw == 250.0
    assert plan.it_load_mw == 187.5
    assert plan.it_energy_mwh == 4500.0
    assert plan.facility_energy_mwh > plan.it_energy_mwh
    assert plan.evaporation_ml > 0
    assert plan.blowdown_ml > 0
    assert np.isclose(
        plan.external_makeup_ml,
        plan.consumption_ml + plan.return_flow_ml,
        atol=1e-12,
    )
    assert plan.offsite_electricity_water_ml > 0


def test_cooling_modes_and_coc_have_expected_directions() -> None:
    weather = {"temperature_c": 34.0, "relative_humidity": 45.0}
    wet = calculate_data_center_plan(
        component("evaporative"), pd.Timestamp("2030-07-01"), weather
    )
    efficient_config = component("efficient_evaporative")
    efficient_config["cooling"].pop("cycles_of_concentration")
    efficient = calculate_data_center_plan(
        efficient_config, pd.Timestamp("2030-07-01"), weather
    )
    high_coc_config = component("evaporative")
    high_coc_config["cooling"]["cycles_of_concentration"] = 8.0
    high_coc = calculate_data_center_plan(
        high_coc_config, pd.Timestamp("2030-07-01"), weather
    )
    dry = calculate_data_center_plan(
        component("dry"), pd.Timestamp("2030-07-01"), weather
    )
    hybrid_config = component("hybrid")
    hybrid_config["cooling"]["hybrid_full_wet_bulb_c"] = 30.0
    hybrid = calculate_data_center_plan(
        hybrid_config, pd.Timestamp("2030-07-01"), weather
    )
    liquid_air = calculate_data_center_plan(
        component("liquid_to_air"), pd.Timestamp("2030-07-01"), weather
    )
    liquid_water = calculate_data_center_plan(
        component("liquid_to_water"), pd.Timestamp("2030-07-01"), weather
    )
    assert high_coc.blowdown_ml < wet.blowdown_ml
    assert efficient.external_makeup_ml < wet.external_makeup_ml
    assert efficient.pue < wet.pue
    assert dry.external_makeup_ml == 0.0
    assert dry.facility_energy_mwh > wet.facility_energy_mwh
    assert 0 < hybrid.wet_cooling_fraction < 1
    assert liquid_air.external_makeup_ml == 0.0
    assert liquid_water.external_makeup_ml > 0.0


def test_reclaimed_water_quality_can_limit_coc_and_increase_makeup() -> None:
    fixed_config = component("evaporative")
    fixed_config["cooling"]["cycles_of_concentration"] = 8.0
    quality_config = component("evaporative")
    quality_config["cooling"].update({
        "cycles_of_concentration": 8.0,
        "coc_mode": "quality_limited",
        "water_quality_limits": {"TDS": 1800.0},
    })
    quality_config["water_sources"]["reclaimed"]["quality"] = {"TDS": 700.0}
    quality_config["water_sources"]["potable"]["quality"] = {"TDS": 200.0}
    drivers = {"temperature_c": 34.0, "relative_humidity": 45.0}
    fixed = calculate_data_center_plan(fixed_config, pd.Timestamp("2030-07-01"), drivers)
    limited = calculate_data_center_plan(quality_config, pd.Timestamp("2030-07-01"), drivers)
    assert limited.cycles_of_concentration < fixed.cycles_of_concentration
    assert limited.external_makeup_ml > fixed.external_makeup_ml


def test_finalize_preserves_withdrawal_consumption_return_and_storage() -> None:
    plan = calculate_data_center_plan(
        component(),
        pd.Timestamp("2030-07-01"),
        {"temperature_c": 30.0, "relative_humidity": 50.0},
    )
    delivered = plan.external_makeup_ml * 0.8
    result = finalize_data_center_day(
        plan,
        external_withdrawal_ml=delivered,
        reclaimed_water_ml=delivered * 0.7,
        potable_water_ml=delivered * 0.3,
        other_water_ml=0.0,
        storage_start_ml=plan.external_makeup_ml * 0.3,
        storage_capacity_ml=plan.external_makeup_ml * 0.5,
    )
    residual = (
        result["external_withdrawal_ml"]
        - result["consumption_ml"]
        - result["return_flow_ml"]
        - result["storage_change_ml"]
    )
    assert abs(residual) < 1e-10
    assert result["unmet_cooling_water_ml"] == 0.0
    assert result["actual_reclaimed_fraction"] == 0.7
