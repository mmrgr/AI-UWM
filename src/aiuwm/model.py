from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_parameters(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_proxy_inputs(parameters: dict[str, Any]) -> pd.DataFrame:
    """Create deterministic daily proxies where proprietary Oslo series are absent."""
    sim = parameters["simulation"]
    dates = pd.date_range(sim["start"], sim["end"], freq="D")
    rng = np.random.default_rng(sim["random_seed"])
    day = dates.dayofyear.to_numpy()

    seasonal_temp = 6.5 + 10.5 * np.sin(2 * np.pi * (day - 172) / 365.25)
    temperature = seasonal_temp + rng.normal(0, 3.0, len(dates))

    wet_probability = 0.42 + 0.08 * np.cos(2 * np.pi * (day - 20) / 365.25)
    wet = rng.random(len(dates)) < wet_probability
    rainfall = np.where(wet, rng.gamma(shape=1.25, scale=4.2, size=len(dates)), 0.0)

    rows: dict[str, Any] = {
        "date": dates,
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
    }
    for resource in parameters["water_supply"]["resources"]:
        mean_daily = resource["mean_inflow_mcm_year"] * 1000.0 / 365.25
        seasonal = 1.0 + 0.38 * np.sin(2 * np.pi * (day - 105) / 365.25)
        noise = rng.lognormal(mean=-0.5 * 0.25**2, sigma=0.25, size=len(dates))
        raw = np.maximum(mean_daily * seasonal * noise, 0.0)
        # Preserve the paper's published annual mean exactly across the proxy record.
        rows[f"{resource['name'].lower()}_inflow_ml"] = raw * mean_daily / raw.mean()
    return pd.DataFrame(rows)


def _growth_factor(year: int, params: dict[str, Any]) -> float:
    demand = params["demand"]
    if year <= 2030:
        return demand["growth_2011_2030"] ** (year - 2011)
    return (
        demand["growth_2011_2030"] ** 19
        * demand["growth_2031_2040"] ** (year - 2030)
    )


def _resource_supply(
    required_production_ml: float,
    inflows_ml: list[float],
    storages_ml: list[float],
    params: dict[str, Any],
    add_new_resource: bool,
    date: pd.Timestamp,
) -> tuple[float, list[float], float]:
    supplied = 0.0
    next_storage: list[float] = []
    resources = params["water_supply"]["resources"]
    for idx, resource in enumerate(resources):
        available = min(resource["capacity_ml"], storages_ml[idx] + inflows_ml[idx])
        requested = required_production_ml * resource["allocation"]
        release = min(requested, resource["wtw_capacity_ml_day"], available)
        supplied += release
        next_storage.append(min(resource["capacity_ml"], available - release))

    new_supply = 0.0
    new = params["water_supply"]["new_resource"]
    if add_new_resource and date >= pd.Timestamp(new["start"]):
        residual = max(0.0, required_production_ml - supplied)
        new_supply = min(residual, new["wtw_capacity_ml_day"], new["daily_inflow_ml"])
        supplied += new_supply
    return supplied, next_storage, new_supply


def _impact_from_energy(
    electricity_kwh: float, diesel_l: float, factors: dict[str, Any]
) -> tuple[float, float, float]:
    elec = factors["electricity"]
    diesel = factors["diesel"]
    diesel_kg = diesel_l * diesel["density_kg_l"]
    ghg = electricity_kwh * elec["ghg_kg_co2e_kwh"] + diesel_kg * diesel["ghg_kg_co2e_kg"]
    acid = electricity_kwh * elec["acid_kg_so2e_kwh"] + diesel_kg * diesel["acid_kg_so2e_kg"]
    eutro = electricity_kwh * elec["eutro_kg_po4e_kwh"] + diesel_kg * diesel["eutro_kg_po4e_kg"]
    return ghg, acid, eutro


def run_scenario(
    inputs: pd.DataFrame,
    parameters: dict[str, Any],
    scenario: str,
) -> pd.DataFrame:
    """Run the daily mass-balance model for BAU, added_resource, or recycling."""
    if scenario not in {"bau", "added_resource", "recycling"}:
        raise ValueError(f"Unknown scenario: {scenario}")

    city = parameters["city"]
    demand_p = parameters["demand"]
    ws = parameters["water_supply"]
    ww = parameters["wastewater"]
    recycle = parameters["recycling"]
    intensity = parameters["component_intensity"]
    impact = parameters["impact_factors"]

    resource_storage = [r["initial_ml"] for r in ws["resources"]]
    sewer_storage = 0.0
    rwh_storage = 0.0
    gwr_storage = 0.0
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for record in inputs.itertuples(index=False):
        date = pd.Timestamp(record.date)
        growth = _growth_factor(date.year, parameters)
        population = city["households_2011"] * city["occupancy"] * growth
        monthly = demand_p["monthly_coefficients"][date.month - 1]

        domestic = population * demand_p["indoor_l_per_capita_day"] / 1_000_000 * monthly
        industrial = demand_p["industrial_ml_day"] * growth * monthly
        irrigation_active = (date.month == 5 and date.day >= 15) or date.month in (6, 7, 8)
        heat_multiplier = max(0.65, min(1.45, 1.0 + 0.025 * (record.temperature_c - 15)))
        irrigation = (
            demand_p["irrigation_ml_day_active"] * growth * heat_multiplier
            if irrigation_active
            else 0.0
        )
        frost_active = date.month in (11, 12, 1, 2, 3)
        frost = demand_p["frost_tapping_ml_day_active"] if frost_active else 0.0
        unregistered = population * demand_p["unregistered_l_per_capita_day"] / 1_000_000
        total_demand = domestic + industrial + irrigation + frost + unregistered

        area_m2 = city["area_ha"] * 10_000
        effective_rain = max(record.rainfall_mm - max(0.0, 0.10 * record.temperature_c), 0.0)
        impervious_runoff = (
            effective_rain
            / 1000
            * area_m2
            * (city["roof_fraction"] + city["road_pavement_fraction"])
            * city["impervious_runoff_coefficient"]
            / 1000
        )
        pervious_runoff = (
            effective_rain
            / 1000
            * area_m2
            * city["pervious_fraction"]
            * max(0.0, 0.18 - 0.10 * city["pervious_infiltration_coefficient"])
            / 1000
        )
        runoff = impervious_runoff + pervious_runoff

        rwh_delivered = 0.0
        gwr_delivered = 0.0
        rwh_captured = 0.0
        gwr_captured = 0.0
        recycling_on = scenario == "recycling" and date >= pd.Timestamp(recycle["start"])
        if recycling_on:
            household_count = city["households_2011"] * growth
            rwh_capacity = (
                household_count
                * recycle["adoption_fraction"]
                * recycle["rwh_tank_m3_per_household"]
                / 1000
            )
            rwh_collection = impervious_runoff * recycle["adoption_fraction"]
            rwh_captured = min(rwh_collection, max(0.0, rwh_capacity - rwh_storage))
            rwh_storage += rwh_captured
            eligible_rwh = domestic * demand_p["indoor_split"]["toilet"] + industrial + irrigation
            rwh_delivered = min(rwh_storage, eligible_rwh)
            rwh_storage -= rwh_delivered

            grey_share = sum(
                demand_p["indoor_split"][key]
                for key in ("dishwasher", "hand_basin", "washing_machine", "shower")
            )
            grey_collection = (
                domestic * grey_share + 0.5 * frost
            ) * recycle["adoption_fraction"]
            gwr_captured = min(
                grey_collection, max(0.0, recycle["gwr_tank_ml"] - gwr_storage)
            )
            gwr_storage += gwr_captured
            remaining_eligible = max(0.0, eligible_rwh - rwh_delivered)
            gwr_delivered = min(gwr_storage, remaining_eligible)
            gwr_storage -= gwr_delivered

        potable_demand = max(0.0, total_demand - rwh_delivered - gwr_delivered)
        required_production = potable_demand / (1.0 - ws["distribution_leakage_fraction"])
        inflows = [getattr(record, f"{r['name'].lower()}_inflow_ml") for r in ws["resources"]]
        production, resource_storage, new_production = _resource_supply(
            required_production,
            inflows,
            resource_storage,
            parameters,
            add_new_resource=scenario == "added_resource",
            date=date,
        )
        delivered_potable = min(potable_demand, production * (1.0 - ws["distribution_leakage_fraction"]))
        leakage = production - delivered_potable
        delivered_total = delivered_potable + rwh_delivered + gwr_delivered
        unmet = max(0.0, total_demand - delivered_total)

        sanitary_before_reuse = delivered_total * demand_p["return_to_sewer_fraction"]
        sanitary = max(0.0, sanitary_before_reuse - gwr_captured)
        runoff_to_sewer = max(0.0, runoff - rwh_captured)
        sewer_inflow = sanitary + runoff_to_sewer
        sewer_storage += sewer_inflow
        sewer_release = min(
            ww["sewer_daily_capacity_ml"],
            ww["sewer_release_a"] * sewer_storage ** ww["sewer_release_b"],
        )
        sewer_storage -= sewer_release
        sewer_overflow = max(0.0, sewer_storage - ww["sewer_storage_ml"])
        sewer_storage -= sewer_overflow

        wwtw1_in = sewer_release * ww["wwtw1_fraction"]
        wwtw2_in = sewer_release * (1.0 - ww["wwtw1_fraction"])
        wwtw_treated = min(wwtw1_in, ww["wwtw1_capacity_ml_day"]) + min(
            wwtw2_in, ww["wwtw2_capacity_ml_day"]
        )
        wwtw_overflow = max(0.0, wwtw1_in - ww["wwtw1_capacity_ml_day"]) + max(
            0.0, wwtw2_in - ww["wwtw2_capacity_ml_day"]
        )

        # Component energy intensities are from paper Table 5.
        wtw_e = production * 1000 * intensity["wtw"]["electricity_kwh_m3"]
        dist_e = production * 1000 * intensity["distribution"]["electricity_kwh_m3"]
        sewer_e = sewer_release * 1000 * intensity["sewer"]["electricity_kwh_m3"]
        wwtw_e = wwtw_treated * 1000 * intensity["wwtw"]["electricity_kwh_m3"]
        recycle_e = (
            rwh_delivered * 1000 * recycle["rwh_electricity_kwh_m3"]
            + gwr_delivered * 1000 * recycle["gwr_electricity_kwh_m3"]
        )
        electricity = wtw_e + dist_e + sewer_e + wwtw_e + recycle_e
        diesel = (
            production * 1000 * (intensity["wtw"]["diesel_l_m3"] + intensity["distribution"]["diesel_l_m3"])
            + sewer_release * 1000 * intensity["sewer"]["diesel_l_m3"]
            + wwtw_treated * 1000 * intensity["wwtw"]["diesel_l_m3"]
        )
        ghg_energy, acid, eutro_resource = _impact_from_energy(electricity, diesel, impact)
        ghg_ch4 = wwtw_treated * 1000 * impact["wwtw_direct"]["ch4_kg_co2e_m3"]
        ghg_n2o = wwtw_treated * 1000 * impact["wwtw_direct"]["n2o_kg_co2e_m3"]

        pollutant = impact["pollutant"]
        sanitary_fraction = sanitary / max(sanitary_before_reuse, 1e-9)
        cod_in = (
            population * pollutant["sanitary_cod_kg_capita_day"] * sanitary_fraction
            + runoff_to_sewer * pollutant["runoff_cod_mg_l"]
        )
        n_in = (
            population * pollutant["sanitary_n_kg_capita_day"] * sanitary_fraction
            + runoff_to_sewer * pollutant["runoff_n_mg_l"]
        )
        p_in = (
            population * pollutant["sanitary_p_kg_capita_day"] * sanitary_fraction
            + runoff_to_sewer * pollutant["runoff_p_mg_l"]
        )
        overflow_fraction = min(1.0, (sewer_overflow + wwtw_overflow) / max(sewer_inflow, 1e-9))
        cod_out = cod_in * (
            overflow_fraction + (1.0 - overflow_fraction) * (1.0 - ww["removal_fraction"]["cod"])
        )
        n_out = n_in * (
            overflow_fraction + (1.0 - overflow_fraction) * (1.0 - ww["removal_fraction"]["nitrogen"])
        )
        p_out = p_in * (
            overflow_fraction + (1.0 - overflow_fraction) * (1.0 - ww["removal_fraction"]["phosphorus"])
        )
        eutro_pollutant = (
            cod_out * pollutant["cod_to_po4e"]
            + n_out * pollutant["n_to_po4e"]
            + p_out * pollutant["p_to_po4e"]
        )

        variable_cost = (
            production * 1000 * intensity["wtw"]["chemical_eur_m3"]
            + wwtw_treated * 1000 * intensity["wwtw"]["chemical_eur_m3"]
        )
        fixed_cost = sum(v.get("fixed_eur_year", 0.0) for v in intensity.values()) / 365.25
        if recycling_on:
            fixed_cost += (
                city["households_2011"]
                * growth
                * recycle["adoption_fraction"]
                * recycle["rwh_fixed_eur_household_year"]
                + recycle["gwr_fixed_eur_year"]
            ) / 365.25

        rows.append(
            {
                "date": date,
                "scenario": scenario,
                "population": population,
                "temperature_c": record.temperature_c,
                "rainfall_mm": record.rainfall_mm,
                "water_demand_ml": total_demand,
                "potable_demand_ml": potable_demand,
                "production_ml": production,
                "new_resource_production_ml": new_production,
                "delivered_potable_ml": delivered_potable,
                "rwh_delivered_ml": rwh_delivered,
                "gwr_delivered_ml": gwr_delivered,
                "delivered_total_ml": delivered_total,
                "unmet_ml": unmet,
                "leakage_ml": leakage,
                "runoff_ml": runoff,
                "runoff_to_sewer_ml": runoff_to_sewer,
                "sanitary_sewage_ml": sanitary,
                "sewer_inflow_ml": sewer_inflow,
                "sewer_outflow_ml": sewer_release,
                "sewer_storage_ml": sewer_storage,
                "sewer_overflow_ml": sewer_overflow,
                "wwtw_treated_ml": wwtw_treated,
                "wwtw_overflow_ml": wwtw_overflow,
                "electricity_kwh": electricity,
                "diesel_l": diesel,
                "ghg_energy_kg_co2e": ghg_energy,
                "ghg_ch4_kg_co2e": ghg_ch4,
                "ghg_n2o_kg_co2e": ghg_n2o,
                "ghg_total_kg_co2e": ghg_energy + ghg_ch4 + ghg_n2o,
                "acidification_kg_so2e": acid,
                "eutrophication_resource_kg_po4e": eutro_resource,
                "eutrophication_pollutant_kg_po4e": eutro_pollutant,
                "eutrophication_total_kg_po4e": eutro_resource + eutro_pollutant,
                "operational_cost_eur": fixed_cost + variable_cost,
                "wr1_storage_ml": resource_storage[0],
                "wr2_storage_ml": resource_storage[1],
            }
        )

    return pd.DataFrame(rows)


def aggregate_results(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = daily.set_index("date")
    numeric = indexed.select_dtypes(include=[np.number])
    monthly = numeric.resample("MS").sum()
    for column in ("population", "temperature_c"):
        monthly[column] = numeric[column].resample("MS").mean()
    annual = numeric.resample("YS").sum()
    for column in ("population", "temperature_c"):
        annual[column] = numeric[column].resample("YS").mean()
    annual["delivered_percent"] = 100 * annual["delivered_total_ml"] / annual["water_demand_ml"]
    annual["ghg_kg_co2e_per_capita"] = annual["ghg_total_kg_co2e"] / annual["population"]

    summary = pd.DataFrame(
        {
            "scenario": [daily["scenario"].iloc[0]],
            "delivered_percent": [100 * daily["delivered_total_ml"].sum() / daily["water_demand_ml"].sum()],
            "unmet_fraction_final_year": [
                annual["unmet_ml"].iloc[-1] / annual["water_demand_ml"].iloc[-1]
            ],
            "mean_annual_ghg_t_co2e": [annual["ghg_total_kg_co2e"].mean() / 1000],
            "mean_annual_acid_t_so2e": [annual["acidification_kg_so2e"].mean() / 1000],
            "mean_annual_eutro_t_po4e": [annual["eutrophication_total_kg_po4e"].mean() / 1000],
            "mean_annual_cost_million_eur": [annual["operational_cost_eur"].mean() / 1e6],
        }
    )
    return monthly.reset_index(), annual.reset_index(), summary
