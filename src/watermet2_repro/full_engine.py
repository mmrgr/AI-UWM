from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .validation import ProjectValidationError, prepare_project, validate_project


WATER_METRICS = (
    "inflow_ml",
    "outflow_ml",
    "delivered_ml",
    "undelivered_ml",
    "loss_ml",
    "leakage_ml",
    "overflow_ml",
    "storage_ml",
    "treated_ml",
    "untreated_ml",
    "infiltration_ml",
    "exfiltration_ml",
    "cso_ml",
    "sto_ml",
    "excess_wastewater_ml",
    "excess_flow_ml",
    "imported_water_ml",
    "exported_water_ml",
    "aquifer_recharge_ml",
    "water_quality_weighted_ml",
)

IMPACT_METRICS = (
    "electricity_kwh",
    "fossil_energy_kwh",
    "embodied_energy_kwh",
    "energy_generated_kwh",
    "ghg_caused_kg_co2e",
    "ghg_avoided_kg_co2e",
    "electricity_ghg_kg_co2e",
    "fossil_ghg_kg_co2e",
    "embodied_ghg_kg_co2e",
    "direct_ghg_kg_co2e",
    "acidification_caused_kg_so2e",
    "acidification_avoided_kg_so2e",
    "electricity_acidification_kg_so2e",
    "fossil_acidification_kg_so2e",
    "embodied_acidification_kg_so2e",
    "eutrophication_caused_kg_po4e",
    "eutrophication_avoided_kg_po4e",
    "electricity_eutrophication_kg_po4e",
    "fossil_eutrophication_kg_po4e",
    "embodied_eutrophication_kg_po4e",
    "operational_cost_eur",
    "capital_cost_eur",
    "maintenance_cost_eur",
    "failure_repair_cost_eur",
    "annualized_capital_cost_eur",
    "expected_failures",
    "expected_outage_hours",
    "sludge_kg",
)


def load_project(project_path: str | Path, timeseries_path: str | Path | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    project_path = Path(project_path)
    project = json.loads(project_path.read_text(encoding="utf-8"))
    if project.get("database_file"):
        database_path = project_path.parent / project["database_file"]
        database = json.loads(database_path.read_text(encoding="utf-8"))
        for key, value in database.items():
            project.setdefault(key, value)
    project = prepare_project(project)
    if timeseries_path is None:
        timeseries_path = project_path.parent / project.get("timeseries_file", "timeseries.csv")
    timeseries = pd.read_csv(timeseries_path, parse_dates=["date"])
    validate_project(project, timeseries)
    start = pd.Timestamp(project["simulation"]["start"])
    end = pd.Timestamp(project["simulation"]["end"])
    timeseries = timeseries[(timeseries["date"] >= start) & (timeseries["date"] <= end)].copy()
    expected = pd.date_range(start, end, freq="D")
    if not timeseries["date"].reset_index(drop=True).equals(pd.Series(expected, name="date")):
        raise ProjectValidationError("timeseries 日期范围必须完整覆盖 simulation.start 至 simulation.end")
    return project, timeseries


def _zero_component_metrics() -> defaultdict[str, float]:
    values: defaultdict[str, float] = defaultdict(float)
    for name in WATER_METRICS + IMPACT_METRICS:
        values[name] = 0.0
    return values


def _active_on(date: pd.Timestamp, start_mmdd: str | None, end_mmdd: str | None) -> bool:
    if not start_mmdd or not end_mmdd:
        return True
    current = date.strftime("%m-%d")
    if start_mmdd <= end_mmdd:
        return start_mmdd <= current <= end_mmdd
    return current >= start_mmdd or current <= end_mmdd


def _deep_set(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target: Any = root
    for part in parts[:-1]:
        if part not in target:
            raise ProjectValidationError(f"干预参数路径不存在: {dotted_path}")
        target = target[part]
    if parts[-1] not in target:
        raise ProjectValidationError(f"干预参数路径不存在: {dotted_path}")
    target[parts[-1]] = value


def _split_mass(masses: dict[str, float], fraction: float) -> dict[str, float]:
    return {name: amount * fraction for name, amount in masses.items()}


def _aggregate_frame(
    frame: pd.DataFrame, frequency: str, identifiers: list[str]
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    numeric = working.select_dtypes(include=[np.number]).columns.tolist()
    state_columns = [
        column for column in ("storage_ml", "population")
        if column in numeric
    ]
    ratio_columns = [
        column
        for column in (
            "delivered_percent",
            "water_quality_index",
            "tap_water_quality_index",
        )
        if column in numeric
    ]
    sum_columns = [
        column for column in numeric
        if column not in state_columns + ratio_columns
    ]
    groupers: list[Any] = [pd.Grouper(key="date", freq=frequency), *identifiers]
    aggregated = working.groupby(groupers, dropna=False)[sum_columns].sum().reset_index()
    if state_columns:
        states = (
            working.groupby(groupers, dropna=False)[state_columns]
            .last()
            .reset_index()
        )
        aggregated = aggregated.merge(states, on=["date", *identifiers], how="left")
    if "water_demand_ml" in aggregated and "delivered_total_ml" in aggregated:
        aggregated["delivered_percent"] = np.where(
            aggregated["water_demand_ml"] > 0,
            100 * aggregated["delivered_total_ml"] / aggregated["water_demand_ml"],
            100.0,
        )
    if "water_quality_weighted_ml" in aggregated and "outflow_ml" in aggregated:
        aggregated["water_quality_index"] = np.where(
            aggregated["outflow_ml"] > 0,
            aggregated["water_quality_weighted_ml"] / aggregated["outflow_ml"],
            np.nan,
        )
    if "tap_quality_weighted_ml" in aggregated and "tap_quality_flow_ml" in aggregated:
        aggregated["tap_water_quality_index"] = np.where(
            aggregated["tap_quality_flow_ml"] > 0,
            aggregated["tap_quality_weighted_ml"] / aggregated["tap_quality_flow_ml"],
            np.nan,
        )
    return aggregated


@dataclass
class FullModelResult:
    system_daily: pd.DataFrame
    subcatchment_daily: pd.DataFrame
    component_daily: pd.DataFrame
    area_daily: pd.DataFrame
    indoor_daily: pd.DataFrame
    pollutant_daily: pd.DataFrame
    recovery_daily: pd.DataFrame
    material_events: pd.DataFrame
    asset_daily: pd.DataFrame
    flood_daily: pd.DataFrame
    risk_daily: pd.DataFrame
    risk_summary: pd.DataFrame

    def write(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.system_daily.to_csv(output / "system_daily.csv", index=False)
        self.subcatchment_daily.to_csv(output / "subcatchment_daily.csv", index=False)
        self.component_daily.to_csv(output / "component_daily.csv", index=False)
        self.area_daily.to_csv(output / "area_daily.csv", index=False)
        self.indoor_daily.to_csv(output / "indoor_daily.csv", index=False)
        self.pollutant_daily.to_csv(output / "pollutant_daily.csv", index=False)
        self.recovery_daily.to_csv(output / "recovery_daily.csv", index=False)
        self.material_events.to_csv(output / "material_events.csv", index=False)
        self.asset_daily.to_csv(output / "asset_daily.csv", index=False)
        self.flood_daily.to_csv(output / "flood_daily.csv", index=False)
        self.risk_daily.to_csv(output / "risk_daily.csv", index=False)
        self.risk_summary.to_csv(output / "risk_summary.csv", index=False)
        tables = {
            "system": (self.system_daily, []),
            "subcatchment": (self.subcatchment_daily, ["subcatchment_id"]),
            "area": (self.area_daily, ["area_id"]),
            "indoor": (self.indoor_daily, ["indoor_id", "local_area"]),
            "component": (self.component_daily, ["component_id", "kind"]),
            "pollutant": (
                self.pollutant_daily,
                ["component_id", "stream", "pollutant"],
            ),
            "recovery": (
                self.recovery_daily,
                ["component_id", "product", "unit"],
            ),
            "asset": (self.asset_daily, ["asset_id", "component_id"]),
            "flood": (self.flood_daily, ["component_id"]),
            "risk": (self.risk_daily, ["risk_code"]),
        }
        for name, (frame, identifiers) in tables.items():
            if frame.empty:
                continue
            _aggregate_frame(frame, "W-MON", identifiers).to_csv(
                output / f"{name}_weekly.csv", index=False
            )
            _aggregate_frame(frame, "MS", identifiers).to_csv(
                output / f"{name}_monthly.csv", index=False
            )
            _aggregate_frame(frame, "YS", identifiers).to_csv(
                output / f"{name}_annual.csv", index=False
            )


@dataclass
class FullWaterMet2Model:
    project: dict[str, Any]
    timeseries: pd.DataFrame
    storage: dict[str, float] = field(default_factory=dict)
    pollutant_storage: dict[str, defaultdict[str, float]] = field(default_factory=dict)
    snowpack_mm: dict[str, float] = field(default_factory=dict)
    previous_snow_depth_mm: dict[str, float] = field(default_factory=dict)
    pipeline_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.project = prepare_project(self.project)
        validate_project(self.project, self.timeseries)
        self.timeseries = self.timeseries.copy()
        for component_id, component in self.project["components"].items():
            if component["kind"] in {"water_resource", "service_reservoir", "sewer", "wwtw", "reuse"}:
                self.storage[component_id] = float(component.get("initial_ml", 0.0))
        self.snowpack_mm = {area_id: 0.0 for area_id in self.project["local_areas"]}
        self.previous_snow_depth_mm = {
            area_id: np.nan for area_id in self.project["local_areas"]
        }
        self.pollutant_storage = {
            component_id: defaultdict(float)
            for component_id, component in self.project["components"].items()
            if component["kind"] in {"sewer", "wwtw", "reuse"}
        }
        self.pipeline_state = {
            pipe["id"]: copy.deepcopy(pipe) for pipe in self.project.get("pipelines", [])
        }
        for pipeline in self.pipeline_state.values():
            pipeline.setdefault(
                "reference_age_years", float(pipeline.get("age_years", 0.0))
            )
            pipeline.setdefault(
                "cohorts",
                [{
                    "length_m": float(pipeline.get("length_m", 0.0)),
                    "age_years": float(pipeline.get("age_years", 0.0)),
                    "material": pipeline.get("material"),
                }],
            )
        self._component_rows: list[dict[str, Any]] = []
        self._area_rows: list[dict[str, Any]] = []
        self._indoor_rows: list[dict[str, Any]] = []
        self._system_rows: list[dict[str, Any]] = []
        self._pollutant_rows: list[dict[str, Any]] = []
        self._recovery_rows: list[dict[str, Any]] = []
        self._material_rows: list[dict[str, Any]] = []
        self._interventions = sorted(self.project.get("interventions", []), key=lambda item: item["date"])
        self._applied_interventions: set[str] = set()

    @classmethod
    def from_files(
        cls, project_path: str | Path, timeseries_path: str | Path | None = None
    ) -> "FullWaterMet2Model":
        project, timeseries = load_project(project_path, timeseries_path)
        return cls(project, timeseries)

    def run(self) -> FullModelResult:
        for row in self.timeseries.itertuples(index=False):
            self._run_day(pd.Timestamp(row.date), row)
        result = FullModelResult(
            system_daily=pd.DataFrame(self._system_rows),
            subcatchment_daily=self._aggregate_subcatchments(),
            component_daily=pd.DataFrame(self._component_rows),
            area_daily=pd.DataFrame(self._area_rows),
            indoor_daily=pd.DataFrame(self._indoor_rows),
            pollutant_daily=pd.DataFrame(self._pollutant_rows),
            recovery_daily=pd.DataFrame(self._recovery_rows),
            material_events=pd.DataFrame(self._material_rows),
            asset_daily=pd.DataFrame(),
            flood_daily=pd.DataFrame(),
            risk_daily=pd.DataFrame(),
            risk_summary=pd.DataFrame(),
        )
        from .risk import evaluate_risks

        risk = evaluate_risks(result, self.project, self.timeseries)
        result.asset_daily = risk.asset_daily
        result.flood_daily = risk.flood_daily
        result.risk_daily = risk.risk_daily
        result.risk_summary = risk.risk_summary
        return result

    def _run_day(self, date: pd.Timestamp, row: Any) -> None:
        self._apply_interventions(date)
        metrics = {component_id: _zero_component_metrics() for component_id in self.project["components"]}
        self._age_pipelines(date)
        self._apply_annual_pipeline_rehabilitation(date, metrics)
        capacity_used: defaultdict[str, float] = defaultdict(float)
        area_state = self._calculate_area_inputs(date, row)
        self._process_reuse(date, area_state, metrics)
        self._initialize_water_storages(row, metrics)
        self._supply_potable_water(date, area_state, metrics, capacity_used)
        wastewater_inputs = self._create_wastewater_inputs(area_state)
        self._route_wastewater(date, wastewater_inputs, metrics)
        self._apply_pipeline_events(date, metrics)
        self._apply_asset_costs_and_failures(date, metrics)
        self._calculate_component_impacts(date, metrics)
        self._save_day(date, area_state, metrics)

    def _apply_interventions(self, date: pd.Timestamp) -> None:
        for index, intervention in enumerate(self._interventions):
            key = str(intervention.get("id", index))
            if key in self._applied_interventions or date < pd.Timestamp(intervention["date"]):
                continue
            for patch in intervention.get("set", []):
                _deep_set(self.project, patch["path"], patch["value"])
            self._applied_interventions.add(key)

    def _population(self, area: dict[str, Any], date: pd.Timestamp, row: Any) -> float:
        if area.get("population_column"):
            return float(getattr(row, area["population_column"]))
        base = float(
            area.get(
                "base_population",
                float(area.get("number_of_properties", 0.0))
                * float(area.get("occupancy_people_property", 0.0)),
            )
        )
        rate = float(area.get("annual_population_growth", 0.0))
        start_year = pd.Timestamp(self.project["simulation"]["start"]).year
        return base * (1.0 + rate) ** (date.year - start_year)

    def _demand_value(
        self, profile: dict[str, Any], population: float, date: pd.Timestamp, row: Any,
        weather: dict[str, str],
    ) -> float:
        if not _active_on(date, profile.get("start_mmdd"), profile.get("end_mmdd")):
            return 0.0
        if profile.get("column"):
            value = float(getattr(row, profile["column"]))
        else:
            base = float(profile.get("base_value", 0.0))
            unit = profile.get("unit", "ml_day")
            if unit == "l_capita_day":
                value = base * population / 1_000_000.0
            elif unit == "m3_day":
                value = base / 1000.0
            elif unit == "ml_day":
                value = base
            else:
                raise ProjectValidationError(f"需求 {profile['name']} 使用了未知单位 {unit}")
            start_year = pd.Timestamp(self.project["simulation"]["start"]).year
            value *= (1 + float(profile.get("annual_growth", 0.0))) ** (
                date.year - start_year
            )
        monthly = profile.get("monthly_factors")
        if monthly:
            value *= float(monthly[date.month - 1])
        if profile.get("temperature_sensitivity"):
            temperature = float(getattr(row, weather.get("temperature", "temperature_c")))
            reference = float(profile.get("reference_temperature_c", 15.0))
            factor = 1 + float(profile["temperature_sensitivity"]) * (
                temperature - reference
            )
            lower, upper = profile.get("temperature_factor_bounds", [0.0, 2.0])
            value *= min(float(upper), max(float(lower), factor))
        return max(0.0, value)

    def _calculate_area_inputs(self, date: pd.Timestamp, row: Any) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for area_id, area in self.project["local_areas"].items():
            indoor_entities = {
                indoor_id: indoor
                for indoor_id, indoor in self.project.get("indoor_areas", {}).items()
                if indoor.get("local_area") == area_id
            }
            indoor_populations = {
                indoor_id: self._population(indoor, date, row)
                for indoor_id, indoor in indoor_entities.items()
            }
            population = (
                sum(indoor_populations.values())
                if indoor_entities else self._population(area, date, row)
            )
            demands: defaultdict[str, float] = defaultdict(float)
            returned: defaultdict[str, float] = defaultdict(float)
            profile_pollutants: defaultdict[str, float] = defaultdict(float)
            appliance_electricity = 0.0
            appliance_operational_cost = 0.0
            appliance_capital_cost = 0.0
            appliance_annualized_capital_cost = 0.0
            indoor_demands: dict[str, dict[str, float]] = {}
            weather = area.get("weather_columns", {})
            profile_groups = [(None, area.get("demand_profiles", []), population)]
            profile_groups.extend(
                (indoor_id, indoor.get("demand_profiles", []), indoor_populations[indoor_id])
                for indoor_id, indoor in indoor_entities.items()
            )
            for indoor_id, profiles, profile_population in profile_groups:
                group_demands: defaultdict[str, float] = defaultdict(float)
                for profile in profiles:
                    name = profile["name"]
                    value = self._demand_value(
                        profile, profile_population, date, row, weather
                    )
                    demands[name] += value
                    returned[name] += value * float(profile.get("return_fraction", 0.95))
                    group_demands[name] += value
                    appliance_electricity += value * 1000.0 * float(
                        profile.get("electricity_kwh_m3", 0.0)
                    )
                    appliance_operational_cost += float(
                        profile.get("maintenance_cost_eur_year", 0.0)
                    ) / 365.25
                    capital = float(profile.get("capital_cost_eur", 0.0))
                    investment = pd.Timestamp(
                        profile.get("investment_date", self.project["simulation"]["start"])
                    )
                    lifetime = float(profile.get("lifetime_years", 0.0))
                    if date == investment or (
                        lifetime > 0
                        and date.month == investment.month
                        and date.day == investment.day
                        and date.year > investment.year
                        and (date.year - investment.year) % max(1, int(round(lifetime))) == 0
                    ):
                        appliance_capital_cost += capital
                    if lifetime > 0:
                        rate = float(self.project.get("finance", {}).get("discount_rate", 0.0))
                        crf = (
                            rate * (1 + rate) ** lifetime
                            / ((1 + rate) ** lifetime - 1)
                            if rate else 1.0 / lifetime
                        )
                        appliance_annualized_capital_cost += capital * crf / 365.25
                    for pollutant, concentration in profile.get(
                        "pollutant_mg_l", {}
                    ).items():
                        profile_pollutants[pollutant] += value * float(concentration)
                if indoor_id:
                    indoor_demands[indoor_id] = dict(group_demands)
            return_fractions = {
                name: returned[name] / value if value else 0.0
                for name, value in demands.items()
            }

            (
                runoff,
                runoff_pollutants,
                surface_runoff,
                surface_pollutants,
                aquifer_recharge,
            ) = self._rainfall_runoff(area_id, area, date, row)
            pollutants = self._sanitary_pollutants(
                area, population, dict(demands), indoor_entities, indoor_populations
            )
            for pollutant, mass in profile_pollutants.items():
                pollutants[pollutant] = pollutants.get(pollutant, 0.0) + mass
            grey_categories = set(area.get("greywater_categories", []))
            total_return = sum(
                value * return_fractions.get(name, 0.0)
                for name, value in demands.items()
            )
            grey_return = sum(
                value * return_fractions.get(name, 0.0)
                for name, value in demands.items() if name in grey_categories
            )
            grey_fraction = grey_return / max(total_return, 1e-12)
            grey_pollutants = {
                pollutant: mass * grey_fraction for pollutant, mass in pollutants.items()
            }
            for pollutant, load in area.get(
                "greywater_pollutant_load_kg_capita_day", {}
            ).items():
                grey_pollutants[pollutant] = population * float(load)
            states[area_id] = {
                "population": population,
                "demand": dict(demands),
                "remaining": dict(demands),
                "return_fractions": return_fractions,
                "runoff_ml": runoff,
                "runoff_remaining_ml": runoff,
                "sanitary_pollutants": pollutants,
                "greywater_pollutants": grey_pollutants,
                "runoff_pollutants": runoff_pollutants,
                "runoff_remaining_pollutants": runoff_pollutants.copy(),
                "surface_runoff_remaining_ml": surface_runoff,
                "surface_runoff_remaining_pollutants": surface_pollutants,
                "aquifer_recharge_ml": aquifer_recharge,
                "appliance_electricity_kwh": appliance_electricity,
                "appliance_operational_cost_eur": appliance_operational_cost,
                "appliance_capital_cost_eur": appliance_capital_cost,
                "appliance_annualized_capital_cost_eur": appliance_annualized_capital_cost,
                "reuse_delivered": defaultdict(float),
                "potable_delivered": defaultdict(float),
                "unmet": defaultdict(float),
                "grey_available_ml": self._greywater_available(area, demands),
                "grey_captured_ml": 0.0,
                "indoor_demands": indoor_demands,
                "indoor_populations": indoor_populations,
            }
        return states

    def _rainfall_runoff(
        self, area_id: str, area: dict[str, Any], date: pd.Timestamp, row: Any
    ) -> tuple[
        float, dict[str, float], dict[str, float], dict[str, dict[str, float]], float
    ]:
        weather = area.get("weather_columns", {})
        precipitation = float(getattr(row, weather.get("precipitation", "rainfall_mm")))
        temperature = float(getattr(row, weather.get("temperature", "temperature_c")))
        precipitation_type = None
        if weather.get("precipitation_type"):
            precipitation_type = str(
                getattr(row, weather["precipitation_type"])
            ).strip().lower()
        if precipitation_type in {"snow", "2"}:
            rainfall = 0.0
        elif precipitation_type in {"sleet", "1"}:
            rainfall = precipitation * 0.5
        else:
            rainfall = precipitation

        if weather.get("snowmelt"):
            snowmelt = float(getattr(row, weather["snowmelt"]))
        elif weather.get("snow_depth"):
            current_depth = float(getattr(row, weather["snow_depth"]))
            previous_depth = self.previous_snow_depth_mm[area_id]
            snowmelt = (
                max(0.0, previous_depth - current_depth)
                * float(area.get("snow_gravity", 0.1))
                if np.isfinite(previous_depth) else 0.0
            )
            self.previous_snow_depth_mm[area_id] = current_depth
        else:
            if precipitation_type is None and temperature <= float(
                area.get("snow_temperature_c", 0.0)
            ):
                self.snowpack_mm[area_id] += precipitation
                rainfall = 0.0
                snowmelt = 0.0
            else:
                melt = float(area.get("degree_day_melt_mm_c_day", 2.0)) * temperature
                snowmelt = min(self.snowpack_mm[area_id], max(0.0, melt))
                self.snowpack_mm[area_id] -= snowmelt
                rainfall = precipitation
        if weather.get("evapotranspiration"):
            evaporation = float(getattr(row, weather["evapotranspiration"]))
        elif all(
            weather.get(name)
            for name in ("wind_speed", "sunshine_hours", "relative_humidity")
        ):
            evaporation = self._reference_evaporation_mm(area, weather, date, row)
        else:
            evaporation = max(0.0, float(area.get("evaporation_coefficient", 0.10)) * temperature)
        effective = max(0.0, rainfall + snowmelt - evaporation)
        area_m2 = float(area["area_ha"]) * 10_000.0
        surfaces = area.get("surfaces", {})
        runoff_ml = 0.0
        pollutant_mass: defaultdict[str, float] = defaultdict(float)
        surface_runoff: dict[str, float] = {}
        surface_pollutants: dict[str, dict[str, float]] = {}
        aquifer_recharge_ml = 0.0
        for surface_name, surface in surfaces.items():
            fraction = float(surface.get("fraction", 0.0))
            coefficient = float(surface.get("runoff_coefficient", 0.0))
            volume_ml = effective / 1000.0 * area_m2 * fraction * coefficient / 1000.0
            infiltrated_ml = (
                effective / 1000.0 * area_m2 * fraction / 1000.0 - volume_ml
            )
            aquifer_recharge_ml += max(0.0, infiltrated_ml) * float(
                surface.get("recharge_fraction", area.get("recharge_fraction", 1.0))
            )
            runoff_ml += volume_ml
            surface_runoff[surface_name] = volume_ml
            surface_mass: dict[str, float] = {}
            for pollutant, concentration_mg_l in surface.get("pollutant_emc_mg_l", {}).items():
                mass = volume_ml * float(concentration_mg_l)
                pollutant_mass[pollutant] += mass
                surface_mass[pollutant] = mass
            surface_pollutants[surface_name] = surface_mass
        return (
            runoff_ml,
            dict(pollutant_mass),
            surface_runoff,
            surface_pollutants,
            aquifer_recharge_ml,
        )

    @staticmethod
    def _reference_evaporation_mm(
        area: dict[str, Any], weather: dict[str, str], date: pd.Timestamp, row: Any
    ) -> float:
        """FAO-56 daily Penman-Monteith reference evaporation from WaterMet² climate fields."""
        temperature = float(getattr(row, weather.get("temperature", "temperature_c")))
        wind = max(0.0, float(getattr(row, weather["wind_speed"])))
        sunshine = max(0.0, float(getattr(row, weather["sunshine_hours"])))
        humidity = min(100.0, max(0.0, float(getattr(row, weather["relative_humidity"]))))
        elevation = float(area.get("elevation_m", 0.0))
        latitude = np.radians(float(area.get("latitude_deg", 45.0)))
        day = date.dayofyear
        dr = 1 + 0.033 * np.cos(2 * np.pi * day / 365)
        declination = 0.409 * np.sin(2 * np.pi * day / 365 - 1.39)
        sunset = np.arccos(np.clip(-np.tan(latitude) * np.tan(declination), -1, 1))
        ra = (
            24 * 60 / np.pi * 0.0820 * dr
            * (
                sunset * np.sin(latitude) * np.sin(declination)
                + np.cos(latitude) * np.cos(declination) * np.sin(sunset)
            )
        )
        daylight = 24 / np.pi * sunset
        solar = (0.25 + 0.50 * min(sunshine, daylight) / max(daylight, 1e-12)) * ra
        clear_sky = (0.75 + 2e-5 * elevation) * ra
        saturation = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
        actual = (
            float(getattr(row, weather["vapour_pressure"]))
            if weather.get("vapour_pressure")
            else saturation * humidity / 100.0
        )
        net_shortwave = 0.77 * solar
        kelvin = temperature + 273.16
        net_longwave = (
            4.903e-9 * kelvin**4 * (0.34 - 0.14 * np.sqrt(max(0.0, actual)))
            * (1.35 * min(1.0, solar / max(clear_sky, 1e-12)) - 0.35)
        )
        net_radiation = net_shortwave - net_longwave
        slope = 4098 * saturation / (temperature + 237.3) ** 2
        pressure = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26
        psychrometric = 0.000665 * pressure
        numerator = (
            0.408 * slope * net_radiation
            + psychrometric * 900 / (temperature + 273) * wind
            * max(0.0, saturation - actual)
        )
        denominator = slope + psychrometric * (1 + 0.34 * wind)
        return max(0.0, numerator / max(denominator, 1e-12))

    def _sanitary_pollutants(
        self,
        area: dict[str, Any],
        population: float,
        demands: dict[str, float],
        indoor_entities: dict[str, dict[str, Any]] | None = None,
        indoor_populations: dict[str, float] | None = None,
    ) -> dict[str, float]:
        masses: defaultdict[str, float] = defaultdict(float)
        indoor_entities = indoor_entities or {}
        indoor_populations = indoor_populations or {}
        if indoor_entities:
            default_loads = area.get("sanitary_pollutant_load_kg_capita_day", {})
            for indoor_id, indoor in indoor_entities.items():
                loads = indoor.get("sanitary_pollutant_load_kg_capita_day", default_loads)
                for pollutant, load in loads.items():
                    masses[pollutant] += indoor_populations[indoor_id] * float(load)
        else:
            for pollutant, load in area.get(
                "sanitary_pollutant_load_kg_capita_day", {}
            ).items():
                masses[pollutant] += population * float(load)
        return dict(masses)

    @staticmethod
    def _greywater_available(area: dict[str, Any], demands: dict[str, float]) -> float:
        grey_categories = set(area.get("greywater_categories", []))
        return sum(value for name, value in demands.items() if name in grey_categories)

    def _process_reuse(
        self,
        date: pd.Timestamp,
        area_state: dict[str, dict[str, Any]],
        metrics: dict[str, defaultdict[str, float]],
    ) -> None:
        reuse_components = [
            (component_id, component)
            for component_id, component in self.project["components"].items()
            if component["kind"] == "reuse" and self._component_active(component, date)
        ]
        reuse_components.sort(key=lambda item: int(item[1].get("priority", 100)))
        for component_id, component in reuse_components:
            reuse_type = component.get("reuse_type", "rwh")
            source_areas = component.get("source_areas", list(area_state))
            target_areas = component.get("target_areas", source_areas)
            collection_fraction = float(component.get("collection_fraction", 1.0))
            available_capacity = max(0.0, float(component.get("capacity_ml", np.inf)) - self.storage[component_id])
            captured = 0.0
            captured_mass: defaultdict[str, float] = defaultdict(float)
            if reuse_type == "rwh":
                selected_surfaces = set(component.get("source_surfaces", []))
                candidates: list[tuple[str, str, float]] = []
                for area_id in source_areas:
                    state = area_state[area_id]
                    for surface, volume in state["surface_runoff_remaining_ml"].items():
                        if not selected_surfaces or surface in selected_surfaces:
                            candidates.append((area_id, surface, volume))
                source_total = sum(volume for _area, _surface, volume in candidates)
                captured = min(available_capacity, source_total * collection_fraction)
                capture_fraction = captured / max(source_total, 1e-12)
                for area_id, surface, volume in candidates:
                    state = area_state[area_id]
                    take = volume * capture_fraction
                    state["surface_runoff_remaining_ml"][surface] -= take
                    state["runoff_remaining_ml"] -= take
                    for pollutant, mass in list(
                        state["surface_runoff_remaining_pollutants"][surface].items()
                    ):
                        removed = mass * capture_fraction
                        captured_mass[pollutant] += removed
                        state["surface_runoff_remaining_pollutants"][surface][
                            pollutant
                        ] -= removed
                        state["runoff_remaining_pollutants"][pollutant] -= removed
            elif reuse_type == "gwr":
                source_total = sum(area_state[area_id]["grey_available_ml"] for area_id in source_areas)
                captured = min(available_capacity, source_total * collection_fraction)
                for area_id in source_areas:
                    state = area_state[area_id]
                    share = state["grey_available_ml"] / source_total if source_total else 0.0
                    take = min(state["grey_available_ml"], captured * share)
                    state["grey_captured_ml"] += take
                    take_fraction = take / max(state["grey_available_ml"], 1e-12)
                    for pollutant, mass in state["greywater_pollutants"].items():
                        captured_mass[pollutant] += mass * take_fraction
            elif reuse_type == "central":
                captured = 0.0  # WWTW effluent is added after wastewater treatment.
            else:
                raise ProjectValidationError(f"回用组件 {component_id} reuse_type={reuse_type!r} 未知")
            self.storage[component_id] += captured
            for pollutant, mass in captured_mass.items():
                self.pollutant_storage[component_id][pollutant] += mass
            metrics[component_id]["inflow_ml"] += captured

            treatment_capacity = float(component.get("treatment_capacity_ml_day", np.inf))
            available = min(self.storage[component_id], treatment_capacity)
            delivered = 0.0
            eligible = component.get("eligible_demands", [])
            for area_id in target_areas:
                state = area_state[area_id]
                for category in eligible:
                    if available <= 0:
                        break
                    requested = state["remaining"].get(category, 0.0)
                    amount = min(requested, available)
                    state["remaining"][category] = requested - amount
                    state["reuse_delivered"][(reuse_type, category)] += amount
                    delivered += amount
                    available -= amount
            storage_before_delivery = self.storage[component_id]
            delivered_fraction = delivered / max(storage_before_delivery, 1e-12)
            removed_mass: dict[str, float] = {}
            delivered_mass: dict[str, float] = {}
            removal = component.get("pollutant_removal_fraction", {})
            for pollutant, stored_mass in list(self.pollutant_storage[component_id].items()):
                treated_mass = stored_mass * delivered_fraction
                removed_mass[pollutant] = treated_mass * float(removal.get(pollutant, 0.0))
                delivered_mass[pollutant] = treated_mass - removed_mass[pollutant]
                self.pollutant_storage[component_id][pollutant] -= treated_mass
            self.storage[component_id] -= delivered
            metrics[component_id]["outflow_ml"] += delivered
            metrics[component_id]["delivered_ml"] += delivered
            metrics[component_id]["treated_ml"] += delivered
            metrics[component_id]["storage_ml"] = self.storage[component_id]
            metrics[component_id]["sludge_kg"] += sum(removed_mass.values())
            self._record_pollutants(date, component_id, "recycled_outflow", delivered_mass)
            self._record_pollutants(date, component_id, "removed_to_sludge", removed_mass)
            self._activity_impacts(component_id, component, delivered, metrics[component_id])

    @staticmethod
    def _component_active(component: dict[str, Any], date: pd.Timestamp) -> bool:
        start = pd.Timestamp(component.get("start_date", "1900-01-01"))
        end = pd.Timestamp(component.get("end_date", "2200-01-01"))
        return start <= date <= end

    def _initialize_water_storages(
        self, row: Any, metrics: dict[str, defaultdict[str, float]]
    ) -> None:
        for component_id, component in self.project["components"].items():
            if component["kind"] != "water_resource":
                continue
            if component.get("resource_type", "surface") in {
                "groundwater", "desalination", "imported"
            }:
                self.storage[component_id] = 0.0
                metrics[component_id]["storage_ml"] = 0.0
                continue
            inflow = float(getattr(row, component["inflow_column"])) if component.get("inflow_column") else float(component.get("constant_inflow_ml_day", 0.0))
            capacity = float(component.get("capacity_ml", np.inf))
            before = self.storage[component_id]
            loss = min(before + inflow, float(component.get("daily_storage_loss_ml", 0.0)))
            available = before + inflow - loss
            overflow = max(0.0, available - capacity)
            self.storage[component_id] = min(capacity, available)
            metrics[component_id]["inflow_ml"] += inflow
            metrics[component_id]["loss_ml"] += loss
            metrics[component_id]["overflow_ml"] += overflow
            metrics[component_id]["storage_ml"] = self.storage[component_id]

    def _supply_potable_water(
        self,
        date: pd.Timestamp,
        area_state: dict[str, dict[str, Any]],
        metrics: dict[str, defaultdict[str, float]],
        capacity_used: defaultdict[str, float],
    ) -> None:
        paths = sorted(self.project["supply_paths"], key=lambda item: int(item.get("priority", 100)))
        potable_requests = {
            area_id: sum(state["remaining"].values()) for area_id, state in area_state.items()
        }
        for path in paths:
            area_id = path["local_area"]
            state = area_state[area_id]
            path_request = potable_requests[area_id] * float(path["allocation"])
            delivered = self._run_supply_path(
                date, path, path_request, metrics, capacity_used
            )
            self._allocate_potable_to_demands(state, delivered, path.get("demand_priority"))

        for area_id, state in area_state.items():
            for category, amount in state["remaining"].items():
                state["unmet"][category] += amount

    def _run_supply_path(
        self,
        date: pd.Timestamp,
        path: dict[str, Any],
        requested_delivery: float,
        metrics: dict[str, defaultdict[str, float]],
        capacity_used: defaultdict[str, float],
    ) -> float:
        chain = path["chain"]
        efficiencies = [1.0 - float(self.project["components"][component_id].get("loss_fraction", self.project["components"][component_id].get("leakage_fraction", 0.0))) for component_id in chain]
        downstream_efficiency: list[float] = []
        product = 1.0
        for efficiency in reversed(efficiencies):
            downstream_efficiency.append(product)
            product *= efficiency
        downstream_efficiency.reverse()
        flow = 0.0
        quality_index = float(
            self.project["components"][chain[0]].get("raw_water_quality_index", 1.0)
        )
        for index, component_id in enumerate(chain):
            component = self.project["components"][component_id]
            kind = component["kind"]
            desired_input = requested_delivery / max(1e-12, efficiencies[index] * downstream_efficiency[index])
            daily_capacity = float(component.get("daily_capacity_ml", component.get("capacity_ml_day", np.inf)))
            capacity_left = max(0.0, daily_capacity - capacity_used[component_id])
            if kind == "water_resource":
                abstraction_capacity = float(component.get("abstraction_capacity_ml_day", daily_capacity))
                capacity_left = min(capacity_left, max(0.0, abstraction_capacity - capacity_used[component_id]))
                if component.get("resource_type", "surface") in {
                    "groundwater", "desalination", "imported"
                }:
                    input_flow = min(desired_input, capacity_left)
                    metrics[component_id]["inflow_ml"] += input_flow
                    if component.get("resource_type") == "imported":
                        metrics[component_id]["imported_water_ml"] += input_flow
                else:
                    input_flow = min(desired_input, self.storage[component_id], capacity_left)
                    self.storage[component_id] -= input_flow
                metrics[component_id]["storage_ml"] = self.storage[component_id]
            elif kind == "service_reservoir":
                capacity = float(component.get("capacity_ml", np.inf))
                admitted = min(flow, max(0.0, capacity - self.storage[component_id]))
                rejected = max(0.0, flow - admitted)
                self.storage[component_id] += admitted
                metrics[component_id]["inflow_ml"] += admitted
                metrics[component_id]["overflow_ml"] += rejected
                input_flow = min(desired_input, self.storage[component_id], capacity_left)
                self.storage[component_id] -= input_flow
                metrics[component_id]["storage_ml"] = self.storage[component_id]
            else:
                input_flow = min(flow, capacity_left)
                if flow > capacity_left:
                    metrics[component_id]["undelivered_ml"] += flow - capacity_left
                metrics[component_id]["inflow_ml"] += input_flow
            capacity_used[component_id] += input_flow
            loss_fraction = 1.0 - efficiencies[index]
            loss = input_flow * loss_fraction
            output_flow = input_flow - loss
            export_fraction = float(component.get("export_fraction", 0.0))
            exported = output_flow * export_fraction
            output_flow -= exported
            metrics[component_id]["exported_water_ml"] += exported
            metrics[component_id]["outflow_ml"] += output_flow
            metrics[component_id]["loss_ml"] += loss
            if kind == "wtw":
                metrics[component_id]["treated_ml"] += input_flow
                quality_index = float(
                    component.get("treated_water_quality_index", quality_index)
                )
            elif kind == "service_reservoir":
                storage_fraction = self.storage[component_id] / max(
                    float(component.get("capacity_ml", np.inf)), 1e-12
                )
                quality_index *= 1.0 - float(
                    component.get("low_storage_quality_penalty", 0.0)
                ) * max(0.0, 1.0 - min(1.0, storage_fraction))
            elif kind == "distribution_main":
                configured_age = float(component.get("asset_age_years", 0.0))
                pipeline_ages = [
                    (
                        float(cohort.get("age_years", 0.0)),
                        float(cohort.get("length_m", 0.0)),
                    )
                    for pipeline in self.pipeline_state.values()
                    if pipeline.get("component_id") == component_id
                    for cohort in pipeline.get("cohorts", [])
                ]
                if pipeline_ages:
                    configured_age = sum(age * length for age, length in pipeline_ages) / max(
                        sum(length for _, length in pipeline_ages), 1e-12
                    )
                quality_index *= np.exp(
                    -float(component.get("quality_decay_per_year", 0.0))
                    * configured_age
                )
            quality_index *= float(component.get("quality_multiplier", 1.0))
            quality_index = float(np.clip(quality_index, 0.0, 1.0))
            metrics[component_id]["water_quality_weighted_ml"] += (
                output_flow * quality_index
            )
            if kind in {"supply_conduit", "trunk_main", "distribution_main"}:
                metrics[component_id]["leakage_ml"] += loss
            self._activity_impacts(component_id, component, input_flow, metrics[component_id])
            if kind == "wtw":
                self._wtw_sludge(
                    date, component_id, component, input_flow, metrics[component_id]
                )
            flow = output_flow
        shortfall = max(0.0, requested_delivery - flow)
        if shortfall:
            metrics[chain[-1]]["undelivered_ml"] += shortfall
        metrics[chain[-1]]["delivered_ml"] += flow
        return flow

    def _wtw_sludge(
        self,
        date: pd.Timestamp,
        component_id: str,
        component: dict[str, Any],
        treated_ml: float,
        metric: defaultdict[str, float],
    ) -> None:
        treated_m3 = treated_ml * 1000.0
        removed: dict[str, float] = {}
        tss = treated_ml * float(component.get("raw_water_tss_mg_l", 0.0))
        if tss:
            removed["TSS"] = tss * float(component.get("tss_removal_fraction", 1.0))
        chemical_sludge = 0.0
        yields = component.get("chemical_sludge_yield_kg_per_kg", {})
        for chemical, dose in component.get("chemical_doses_kg_m3", {}).items():
            chemical_sludge += (
                treated_m3 * float(dose) * float(yields.get(chemical, 0.0))
            )
        direct_sludge = treated_m3 * float(component.get("wtw_sludge_kg_m3", 0.0))
        sludge = sum(removed.values()) + chemical_sludge + direct_sludge
        metric["sludge_kg"] += sludge
        if chemical_sludge + direct_sludge:
            removed["treatment_residuals"] = chemical_sludge + direct_sludge
        self._record_pollutants(date, component_id, "removed_to_sludge", removed)

    @staticmethod
    def _allocate_potable_to_demands(
        state: dict[str, Any], delivered: float, priority: Iterable[str] | None
    ) -> None:
        order = list(priority or state["remaining"].keys())
        for category in order:
            requested = state["remaining"].get(category, 0.0)
            amount = min(requested, delivered)
            state["remaining"][category] = requested - amount
            state["potable_delivered"][category] += amount
            delivered -= amount
            if delivered <= 0:
                return

    def _create_wastewater_inputs(
        self, area_state: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        inputs: dict[str, dict[str, Any]] = defaultdict(lambda: {"volume_ml": 0.0, "pollutants": defaultdict(float)})
        for area_id, area in self.project["local_areas"].items():
            state = area_state[area_id]
            delivered_by_category = defaultdict(float)
            for category, amount in state["potable_delivered"].items():
                delivered_by_category[category] += amount
            for (_reuse_type, category), amount in state["reuse_delivered"].items():
                delivered_by_category[category] += amount
            sanitary = sum(
                amount * state["return_fractions"].get(category, 0.95)
                for category, amount in delivered_by_category.items()
            )
            sanitary = max(0.0, sanitary - state["grey_captured_ml"])
            state["sanitary_ml"] = sanitary
            sanitary_sewer = area.get("sanitary_sewer") or area.get("combined_sewer")
            storm_sewer = area.get("storm_sewer") or area.get("combined_sewer")
            if sanitary_sewer:
                inputs[sanitary_sewer]["volume_ml"] += sanitary
                original_return = sum(
                    state["demand"][name] * state["return_fractions"].get(name, 0.95)
                    for name in state["demand"]
                )
                fraction = sanitary / max(original_return, 1e-12)
                for pollutant, mass in state["sanitary_pollutants"].items():
                    inputs[sanitary_sewer]["pollutants"][pollutant] += mass * fraction
            if storm_sewer:
                inputs[storm_sewer]["volume_ml"] += state["runoff_remaining_ml"]
                for pollutant, mass in state["runoff_remaining_pollutants"].items():
                    inputs[storm_sewer]["pollutants"][pollutant] += mass
        return inputs

    def _route_wastewater(
        self,
        date: pd.Timestamp,
        wastewater_inputs: dict[str, dict[str, Any]],
        metrics: dict[str, defaultdict[str, float]],
    ) -> None:
        components = self.project["components"]
        connections = self.project.get("wastewater_connections", [])
        outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for connection in connections:
            outgoing[connection["from"]].append(connection)
        order = self._wastewater_order(outgoing)
        node_inputs: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"volume_ml": 0.0, "pollutants": defaultdict(float)}
        )
        for component_id, value in wastewater_inputs.items():
            node_inputs[component_id]["volume_ml"] += value["volume_ml"]
            for pollutant, mass in value["pollutants"].items():
                node_inputs[component_id]["pollutants"][pollutant] += mass

        for component_id in order:
            component = components[component_id]
            kind = component["kind"]
            incoming = node_inputs[component_id]
            volume = float(incoming["volume_ml"])
            pollutants = dict(incoming["pollutants"])
            metrics[component_id]["inflow_ml"] += volume
            if kind == "sewer":
                outflow, overflow, out_mass, overflow_mass, exfiltration_mass = self._process_sewer(
                    component_id, component, volume, pollutants, metrics[component_id]
                )
                overflow_stream = {
                    "combined": "cso",
                    "storm": "sto",
                    "sanitary": "excess_wastewater",
                }.get(component.get("sewer_type"), "overflow")
                self._record_pollutants(date, component_id, "outflow", out_mass)
                self._record_pollutants(
                    date, component_id, overflow_stream, overflow_mass
                )
                self._record_pollutants(
                    date, component_id, "exfiltration", exfiltration_mass
                )
                self._send_flow(component_id, outflow, out_mass, outgoing, node_inputs)
                self._send_overflow(component, overflow, overflow_mass, node_inputs)
                if component.get("exfiltration_to") or component.get("overflow_to"):
                    exfiltration_target = {
                        **component,
                        "overflow_to": component.get(
                            "exfiltration_to", component.get("overflow_to")
                        ),
                    }
                    self._send_overflow(
                        exfiltration_target,
                        metrics[component_id]["exfiltration_ml"],
                        exfiltration_mass,
                        node_inputs,
                    )
            elif kind == "wwtw":
                treated, untreated, treated_mass, untreated_mass, removed = self._process_wwtw(
                    date, component_id, component, volume, pollutants, metrics
                )
                self._record_pollutants(date, component_id, "treated_outflow", treated_mass)
                self._record_pollutants(date, component_id, "untreated_outflow", untreated_mass)
                self._record_pollutants(date, component_id, "removed_to_sludge", removed)
                self._send_flow(component_id, treated, treated_mass, outgoing, node_inputs)
                self._send_overflow(component, untreated, untreated_mass, node_inputs)
            elif kind == "receiving_water":
                metrics[component_id]["outflow_ml"] += volume
                eutro_factors = self.project.get("characterisation", {}).get(
                    "pollutant_to_po4e", {}
                )
                metrics[component_id]["eutrophication_caused_kg_po4e"] += sum(
                    mass * float(eutro_factors.get(pollutant, 0.0))
                    for pollutant, mass in pollutants.items()
                )
                self._record_pollutants(date, component_id, "received", pollutants)

    def _wastewater_order(self, outgoing: dict[str, list[dict[str, Any]]]) -> list[str]:
        components = self.project["components"]
        nodes = [
            component_id
            for component_id, component in components.items()
            if component["kind"] in {"sewer", "wwtw", "receiving_water"}
        ]
        indegree = {node: 0 for node in nodes}
        ordering_targets: dict[str, set[str]] = defaultdict(set)
        for source, connections in outgoing.items():
            if source not in indegree:
                continue
            for connection in connections:
                target = connection["to"]
                if target in indegree:
                    ordering_targets[source].add(target)
        for source, component in components.items():
            if source not in indegree:
                continue
            for field in ("overflow_to", "exfiltration_to"):
                target = component.get(field)
                if target in indegree:
                    ordering_targets[source].add(target)
        for targets in ordering_targets.values():
            for target in targets:
                indegree[target] += 1
        queue = [node for node in nodes if indegree[node] == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for target in ordering_targets.get(node, set()):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(order) != len(nodes):
            raise ProjectValidationError("污水拓扑包含环；WaterMet² 日路由要求有向无环连接")
        return order

    def _process_sewer(
        self,
        component_id: str,
        component: dict[str, Any],
        inflow: float,
        pollutants: dict[str, float],
        metric: defaultdict[str, float],
    ) -> tuple[
        float, float, dict[str, float], dict[str, float], dict[str, float]
    ]:
        infiltration = inflow * float(component.get("infiltration_fraction", 0.0))
        exfiltration = inflow * float(component.get("exfiltration_fraction", 0.0))
        metric["inflow_ml"] += infiltration
        metric["infiltration_ml"] += infiltration
        metric["exfiltration_ml"] += exfiltration
        previous = self.storage[component_id]
        before_exfiltration = previous + inflow + infiltration
        exfiltration = min(exfiltration, before_exfiltration)
        total = before_exfiltration - exfiltration
        current_mass = dict(self.pollutant_storage[component_id])
        for pollutant, mass in pollutants.items():
            current_mass[pollutant] = current_mass.get(pollutant, 0.0) + mass
        exfiltration_fraction = exfiltration / max(before_exfiltration, 1e-12)
        exfiltration_mass = _split_mass(current_mass, exfiltration_fraction)
        remaining_mass = {
            pollutant: mass - exfiltration_mass.get(pollutant, 0.0)
            for pollutant, mass in current_mass.items()
        }
        mode = component.get("capacity_mode", "storage_release")
        if mode == "storage_release":
            a = float(component.get("release_a", 1.0))
            b = float(component.get("release_b", 1.0))
            outflow = min(total, a * total**b, float(component.get("daily_capacity_ml", np.inf)))
            remaining = total - outflow
            capacity = float(component.get("capacity_ml", np.inf))
            overflow = max(0.0, remaining - capacity)
            self.storage[component_id] = min(capacity, remaining)
        elif mode == "transmission":
            outflow = min(total, float(component.get("daily_capacity_ml", np.inf)))
            overflow = max(0.0, total - outflow)
            self.storage[component_id] = 0.0
        else:
            raise ProjectValidationError(f"污水管网 {component_id} capacity_mode={mode!r} 未知")
        release_fraction = outflow / max(total, 1e-12)
        overflow_fraction = overflow / max(total, 1e-12)
        out_mass = _split_mass(remaining_mass, release_fraction)
        overflow_mass = _split_mass(remaining_mass, overflow_fraction)
        self.pollutant_storage[component_id] = defaultdict(
            float,
            {
                pollutant: max(0.0, mass - out_mass.get(pollutant, 0.0) - overflow_mass.get(pollutant, 0.0))
                for pollutant, mass in remaining_mass.items()
            },
        )
        metric["outflow_ml"] += outflow
        metric["overflow_ml"] += overflow
        overflow_metric = {
            "combined": "cso_ml",
            "storm": "sto_ml",
            "sanitary": "excess_wastewater_ml",
        }.get(component.get("sewer_type"), "excess_flow_ml")
        metric[overflow_metric] += overflow
        metric["loss_ml"] += exfiltration
        metric["storage_ml"] = self.storage[component_id]
        self._activity_impacts(component_id, component, outflow, metric)
        return outflow, overflow, out_mass, overflow_mass, exfiltration_mass

    def _process_wwtw(
        self,
        date: pd.Timestamp,
        component_id: str,
        component: dict[str, Any],
        inflow: float,
        pollutants: dict[str, float],
        metrics: dict[str, defaultdict[str, float]],
    ) -> tuple[float, float, dict[str, float], dict[str, float], dict[str, float]]:
        metric = metrics[component_id]
        previous = self.storage[component_id]
        total = previous + inflow
        treatment = min(total, float(component.get("daily_capacity_ml", np.inf)))
        remaining = total - treatment
        capacity = float(component.get("capacity_ml", 0.0))
        untreated = max(0.0, remaining - capacity)
        self.storage[component_id] = min(capacity, remaining)
        current_mass = dict(self.pollutant_storage[component_id])
        for pollutant, mass in pollutants.items():
            current_mass[pollutant] = current_mass.get(pollutant, 0.0) + mass
        treated_fraction = treatment / max(total, 1e-12)
        untreated_fraction = untreated / max(total, 1e-12)
        treated_mass: dict[str, float] = {}
        untreated_mass = _split_mass(current_mass, untreated_fraction)
        removed: dict[str, float] = {}
        removal = component.get("pollutant_removal_fraction", {})
        for pollutant, mass in current_mass.items():
            mass_to_treatment = mass * treated_fraction
            removed[pollutant] = mass_to_treatment * float(removal.get(pollutant, 0.0))
            treated_mass[pollutant] = mass_to_treatment - removed[pollutant]
        self.pollutant_storage[component_id] = defaultdict(
            float,
            {
                pollutant: max(
                    0.0,
                    mass
                    - treated_mass.get(pollutant, 0.0)
                    - removed.get(pollutant, 0.0)
                    - untreated_mass.get(pollutant, 0.0),
                )
                for pollutant, mass in current_mass.items()
            },
        )
        metric["treated_ml"] += treatment
        metric["outflow_ml"] += treatment
        metric["untreated_ml"] += untreated
        metric["overflow_ml"] += untreated
        metric["storage_ml"] = self.storage[component_id]
        metric["sludge_kg"] += sum(removed.values())
        self._activity_impacts(component_id, component, treatment, metric)
        self._direct_wwtw_impacts(component, treatment, metric)
        self._resource_recovery(date, component_id, component, treatment, removed, metric)
        self._sludge_handling(date, component_id, component, sum(removed.values()), metric)
        central_id = component.get("central_reuse_component")
        if central_id:
            reuse = self.project["components"][central_id]
            fraction = float(component.get("central_reuse_fraction", 0.0))
            transfer = min(treatment * fraction, max(0.0, float(reuse.get("capacity_ml", np.inf)) - self.storage[central_id]))
            self.storage[central_id] += transfer
            metrics[central_id]["inflow_ml"] += transfer
            metrics[central_id]["storage_ml"] = self.storage[central_id]
            transfer_fraction = transfer / max(treatment, 1e-12)
            for pollutant, mass in treated_mass.items():
                self.pollutant_storage[central_id][pollutant] += mass * transfer_fraction
            discharge_fraction = 1.0 - transfer_fraction
            treated_mass = _split_mass(treated_mass, discharge_fraction)
            treatment -= transfer
        return treatment, untreated, treated_mass, untreated_mass, removed

    def _sludge_handling(
        self,
        date: pd.Timestamp,
        component_id: str,
        component: dict[str, Any],
        dry_solids_kg: float,
        metric: defaultdict[str, float],
    ) -> None:
        process = component.get("sludge_process")
        if not process or dry_solids_kg <= 0:
            return
        digested_dry = dry_solids_kg * (
            1.0 - float(process.get("digestion_mass_reduction_fraction", 0.0))
        )
        dewatered_fraction = max(
            float(process.get("dewatered_dry_solids_fraction", 1.0)), 1e-12
        )
        dried_fraction = max(
            float(process.get("dried_dry_solids_fraction", 1.0)), 1e-12
        )
        products = {
            "digested_sludge_dry_solids": digested_dry,
            "dewatered_sludge": digested_dry / dewatered_fraction,
            "dried_sludge": digested_dry / dried_fraction,
            "biosolids_to_end_use": digested_dry
            * float(process.get("end_use_fraction", 1.0)),
        }
        for product, amount in products.items():
            metric[f"recovered_{product}_kg"] += amount
            self._recovery_rows.append(
                {
                    "date": date,
                    "component_id": component_id,
                    "product": product,
                    "amount": amount,
                    "unit": "kg",
                }
            )

    @staticmethod
    def _send_flow(
        source: str,
        volume: float,
        pollutants: dict[str, float],
        outgoing: dict[str, list[dict[str, Any]]],
        node_inputs: dict[str, dict[str, Any]],
    ) -> None:
        connections = outgoing.get(source, [])
        if not connections:
            return
        total_fraction = sum(float(connection.get("fraction", 0.0)) for connection in connections)
        if total_fraction > 1.000001:
            raise ProjectValidationError(f"{source} 的下游 fraction 合计超过 1")
        for connection in connections:
            fraction = float(connection.get("fraction", 0.0))
            target = connection["to"]
            node_inputs[target]["volume_ml"] += volume * fraction
            for pollutant, mass in pollutants.items():
                node_inputs[target]["pollutants"][pollutant] += mass * fraction

    def _send_overflow(
        self,
        component: dict[str, Any],
        volume: float,
        pollutants: dict[str, float],
        node_inputs: dict[str, dict[str, Any]],
    ) -> None:
        target = component.get("overflow_to")
        if not target or volume <= 0:
            return
        node_inputs[target]["volume_ml"] += volume
        for pollutant, mass in pollutants.items():
            node_inputs[target]["pollutants"][pollutant] += mass

    def _record_pollutants(
        self, date: pd.Timestamp, component_id: str, stream: str, pollutants: dict[str, float]
    ) -> None:
        for pollutant, mass in pollutants.items():
            self._pollutant_rows.append(
                {
                    "date": date,
                    "component_id": component_id,
                    "stream": stream,
                    "pollutant": pollutant,
                    "mass_kg": mass,
                }
            )

    def _activity_impacts(
        self,
        component_id: str,
        component: dict[str, Any],
        activity_ml: float,
        metric: defaultdict[str, float],
    ) -> None:
        activity_m3 = activity_ml * 1000.0
        electricity = activity_m3 * float(component.get("electricity_kwh_m3", 0.0))
        metric["electricity_kwh"] += electricity
        energy_sources = self.project.get("energy_sources", {})
        electricity_factor = energy_sources.get("electricity", {})
        electricity_ghg = electricity * float(electricity_factor.get("ghg_kg_co2e_unit", 0.0))
        electricity_acid = electricity * float(electricity_factor.get("acid_kg_so2e_unit", 0.0))
        electricity_eutro = electricity * float(electricity_factor.get("eutro_kg_po4e_unit", 0.0))
        metric["ghg_caused_kg_co2e"] += electricity_ghg
        metric["electricity_ghg_kg_co2e"] += electricity_ghg
        metric["acidification_caused_kg_so2e"] += electricity_acid
        metric["electricity_acidification_kg_so2e"] += electricity_acid
        metric["eutrophication_caused_kg_po4e"] += electricity_eutro
        metric["electricity_eutrophication_kg_po4e"] += electricity_eutro
        metric["operational_cost_eur"] += electricity * float(electricity_factor.get("cost_eur_unit", 0.0))

        for source_name, amount_per_m3 in component.get("fossil_fuels", {}).items():
            source = energy_sources.get(source_name, {})
            amount = activity_m3 * float(amount_per_m3)
            energy = amount * float(source.get("energy_kwh_unit", 0.0))
            metric["fossil_energy_kwh"] += energy
            fossil_ghg = amount * float(source.get("ghg_kg_co2e_unit", 0.0))
            fossil_acid = amount * float(source.get("acid_kg_so2e_unit", 0.0))
            fossil_eutro = amount * float(source.get("eutro_kg_po4e_unit", 0.0))
            metric["ghg_caused_kg_co2e"] += fossil_ghg
            metric["fossil_ghg_kg_co2e"] += fossil_ghg
            metric["acidification_caused_kg_so2e"] += fossil_acid
            metric["fossil_acidification_kg_so2e"] += fossil_acid
            metric["eutrophication_caused_kg_po4e"] += fossil_eutro
            metric["fossil_eutrophication_kg_po4e"] += fossil_eutro
            metric["operational_cost_eur"] += amount * float(source.get("cost_eur_unit", 0.0))

        chemicals = self.project.get("chemicals", {})
        for chemical_name, dose_kg_m3 in component.get("chemical_doses_kg_m3", {}).items():
            chemical = chemicals.get(chemical_name)
            if chemical is None:
                raise ProjectValidationError(f"组件 {component_id} 使用了未定义化学品 {chemical_name}")
            amount = activity_m3 * float(dose_kg_m3)
            metric[f"chemical_{chemical_name}_kg"] += amount
            metric["embodied_energy_kwh"] += amount * float(chemical.get("embodied_energy_kwh_kg", 0.0))
            embodied_ghg = amount * float(chemical.get("ghg_kg_co2e_kg", 0.0))
            embodied_acid = amount * float(chemical.get("acid_kg_so2e_kg", 0.0))
            embodied_eutro = amount * float(chemical.get("eutro_kg_po4e_kg", 0.0))
            metric["ghg_caused_kg_co2e"] += embodied_ghg
            metric["embodied_ghg_kg_co2e"] += embodied_ghg
            metric["acidification_caused_kg_so2e"] += embodied_acid
            metric["embodied_acidification_kg_so2e"] += embodied_acid
            metric["eutrophication_caused_kg_po4e"] += embodied_eutro
            metric["embodied_eutrophication_kg_po4e"] += embodied_eutro
            metric["operational_cost_eur"] += amount * float(chemical.get("cost_eur_kg", 0.0))

        metric["operational_cost_eur"] += activity_m3 * float(component.get("variable_cost_eur_m3", 0.0))

        generated = activity_m3 * float(component.get("energy_generation_kwh_m3", 0.0))
        if component.get("hydraulic_head_m"):
            generated += (
                activity_m3 * 9.81 * float(component["hydraulic_head_m"])
                * float(component.get("turbine_efficiency", 0.75)) / 3600.0
            )
        if generated:
            metric["energy_generated_kwh"] += generated
            metric["ghg_avoided_kg_co2e"] += generated * float(
                electricity_factor.get("ghg_kg_co2e_unit", 0.0)
            )
            metric["acidification_avoided_kg_so2e"] += generated * float(
                electricity_factor.get("acid_kg_so2e_unit", 0.0)
            )
            metric["eutrophication_avoided_kg_po4e"] += generated * float(
                electricity_factor.get("eutro_kg_po4e_unit", 0.0)
            )
            metric["operational_cost_eur"] -= generated * float(
                electricity_factor.get("value_eur_unit", electricity_factor.get("cost_eur_unit", 0.0))
            )

    def _direct_wwtw_impacts(
        self, component: dict[str, Any], treated_ml: float, metric: defaultdict[str, float]
    ) -> None:
        treated_m3 = treated_ml * 1000.0
        direct = component.get("direct_emissions", {})
        gwp = self.project.get("characterisation", {}).get("gwp100", {"co2": 1, "ch4": 25, "n2o": 298})
        ch4 = treated_m3 * float(direct.get("ch4_kg_m3", 0.0))
        n2o = treated_m3 * float(direct.get("n2o_kg_m3", 0.0))
        co2 = treated_m3 * float(direct.get("co2_kg_m3", 0.0))
        metric["ghg_ch4_kg"] += ch4
        metric["ghg_n2o_kg"] += n2o
        metric["ghg_co2_kg"] += co2
        direct_ghg = (
            ch4 * float(gwp.get("ch4", 25))
            + n2o * float(gwp.get("n2o", 298))
            + co2 * float(gwp.get("co2", 1))
        )
        metric["ghg_caused_kg_co2e"] += direct_ghg
        metric["direct_ghg_kg_co2e"] += direct_ghg
        acid_cf = self.project.get("characterisation", {}).get("acidification", {})
        eutro_cf = self.project.get("characterisation", {}).get("eutrophication", {})
        sludge_kg = metric["sludge_kg"]
        sludge_ch4 = sludge_kg * float(direct.get("sludge_ch4_kg_kg", 0.0))
        sludge_n2o = sludge_kg * float(direct.get("sludge_n2o_kg_kg", 0.0))
        metric["ghg_ch4_kg"] += sludge_ch4
        metric["ghg_n2o_kg"] += sludge_n2o
        sludge_ghg = (
            sludge_ch4 * float(gwp.get("ch4", 25))
            + sludge_n2o * float(gwp.get("n2o", 298))
        )
        metric["ghg_caused_kg_co2e"] += sludge_ghg
        metric["direct_ghg_kg_co2e"] += sludge_ghg
        nh3 = treated_m3 * float(direct.get("nh3_kg_m3", 0.0))
        no2 = treated_m3 * float(direct.get("no2_kg_m3", 0.0))
        so2 = treated_m3 * float(direct.get("so2_kg_m3", 0.0))
        metric["acidification_caused_kg_so2e"] += (
            nh3 * float(acid_cf.get("nh3", 2.45))
            + no2 * float(acid_cf.get("no2", 0.56))
            + so2 * float(acid_cf.get("so2", 1.0))
        )
        metric["eutrophication_caused_kg_po4e"] += nh3 * float(eutro_cf.get("nh3", 3.8))

    def _age_pipelines(self, date: pd.Timestamp) -> None:
        """Advance age cohorts and update age-dependent component leakage."""
        simulation_start = pd.Timestamp(self.project["simulation"]["start"])
        if date.month != 1 or date.day != 1 or date == simulation_start:
            return
        for pipeline in self.pipeline_state.values():
            for cohort in pipeline["cohorts"]:
                cohort["age_years"] = float(cohort.get("age_years", 0.0)) + 1.0
        self._update_pipeline_performance()

    def _update_pipeline_performance(self) -> None:
        by_component: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for pipeline in self.pipeline_state.values():
            curve = sorted(
                pipeline.get("leakage_curve", []),
                key=lambda point: float(point["age_years"]),
            )
            base = float(
                pipeline.get(
                    "base_leakage_fraction",
                    self.project["components"][pipeline["component_id"]].get(
                        "leakage_fraction", 0.0
                    ),
                )
            )
            growth = float(pipeline.get("leakage_growth_per_year", 0.0))
            for cohort in pipeline["cohorts"]:
                age = float(cohort.get("age_years", 0.0))
                if curve:
                    leakage = float(np.interp(
                        age,
                        [float(point["age_years"]) for point in curve],
                        [float(point["leakage_fraction"]) for point in curve],
                    ))
                else:
                    reference_age = float(pipeline.get("reference_age_years", 0.0))
                    leakage = base * (1.0 + growth) ** max(0.0, age - reference_age)
                by_component[pipeline["component_id"]].append(
                    (float(cohort["length_m"]), min(1.0, leakage))
                )
        for component_id, values in by_component.items():
            total = sum(length for length, _leakage in values)
            if total:
                self.project["components"][component_id]["leakage_fraction"] = sum(
                    length * leakage for length, leakage in values
                ) / total

    def _apply_annual_pipeline_rehabilitation(
        self, date: pd.Timestamp, metrics: dict[str, defaultdict[str, float]]
    ) -> None:
        simulation_start = pd.Timestamp(self.project["simulation"]["start"])
        if date.month != 1 or date.day != 1 or date == simulation_start:
            return
        for pipeline in self.pipeline_state.values():
            method = pipeline.get("rehabilitation_method")
            if not method:
                continue
            total_length = sum(float(cohort["length_m"]) for cohort in pipeline["cohorts"])
            length = float(
                pipeline.get(
                    "annual_rehabilitation_length_m",
                    total_length * float(pipeline.get("annual_rehabilitation_rate", 0.0)),
                )
            )
            if length > 0:
                self._rehabilitate_pipeline(
                    date, pipeline, method, min(length, total_length), metrics, "annual"
                )

    def _resource_recovery(
        self,
        date: pd.Timestamp,
        component_id: str,
        component: dict[str, Any],
        treated_ml: float,
        removed: dict[str, float],
        metric: defaultdict[str, float],
    ) -> None:
        byproducts = self.project.get("byproducts", {})
        treated_m3 = treated_ml * 1000.0
        for product_name, recovery in component.get("resource_recovery", {}).items():
            if "kg_m3" in recovery:
                amount = treated_m3 * float(recovery["kg_m3"])
                unit = "kg"
            elif "kwh_m3" in recovery:
                amount = treated_m3 * float(recovery["kwh_m3"])
                unit = "kwh"
            elif "from_pollutant" in recovery:
                source = recovery["from_pollutant"]
                amount = removed.get(source, 0.0) * float(recovery.get("yield", 1.0))
                unit = "kg"
            else:
                continue
            self._credit_recovery(
                date, component_id, product_name, amount, unit, metric, byproducts
            )

        process = component.get("process_recovery", {})
        if process:
            biogas = metric["sludge_kg"] * float(
                process.get("biogas_kg_per_kg_sludge", 0.0)
            )
            if biogas:
                self._credit_recovery(
                    date, component_id, "biogas", biogas, "kg", metric, byproducts
                )
            biogas_energy = biogas * float(process.get("biogas_lhv_kwh_kg", 6.0))
            electricity = biogas_energy * float(
                process.get("chp_electric_efficiency", 0.0)
            )
            heat = biogas_energy * float(process.get("chp_heat_efficiency", 0.0))
            heat += (
                treated_m3
                * 1.163
                * float(process.get("effluent_temperature_drop_c", 0.0))
                * float(process.get("heat_recovery_efficiency", 1.0))
            )
            if electricity:
                self._credit_recovery(
                    date, component_id, "generated_electricity", electricity,
                    "kwh", metric, byproducts
                )
            if heat:
                self._credit_recovery(
                    date, component_id, "recovered_heat", heat, "kwh",
                    metric, byproducts
                )

    def _credit_recovery(
        self,
        date: pd.Timestamp,
        component_id: str,
        product_name: str,
        amount: float,
        unit: str,
        metric: defaultdict[str, float],
        byproducts: dict[str, Any],
    ) -> None:
        product = byproducts.get(product_name, {})
        metric[f"recovered_{product_name}_{unit}"] += amount
        if unit == "kwh":
            metric["energy_generated_kwh"] += amount
        metric["embodied_energy_kwh"] -= amount * float(
            product.get("embodied_energy_kwh_unit", 0.0)
        )
        metric["ghg_avoided_kg_co2e"] += amount * float(
            product.get("ghg_kg_co2e_unit", 0.0)
        )
        metric["acidification_avoided_kg_so2e"] += amount * float(
            product.get("acid_kg_so2e_unit", 0.0)
        )
        metric["eutrophication_avoided_kg_po4e"] += amount * float(
            product.get("eutro_kg_po4e_unit", 0.0)
        )
        metric["operational_cost_eur"] -= amount * float(
            product.get("value_eur_unit", 0.0)
        )
        self._recovery_rows.append({
            "date": date,
            "component_id": component_id,
            "product": product_name,
            "amount": amount,
            "unit": unit,
        })

    def _calculate_component_impacts(
        self, date: pd.Timestamp, metrics: dict[str, defaultdict[str, float]]
    ) -> None:
        for component_id, component in self.project["components"].items():
            if not self._component_active(component, date):
                continue
            metrics[component_id]["operational_cost_eur"] += float(component.get("fixed_cost_eur_year", 0.0)) / 365.25
            metrics[component_id]["water_quality_index"] = (
                metrics[component_id]["water_quality_weighted_ml"]
                / max(metrics[component_id]["outflow_ml"], 1e-12)
                if metrics[component_id]["outflow_ml"] > 0
                else np.nan
            )
            metrics[component_id]["ghg_net_kg_co2e"] = metrics[component_id]["ghg_caused_kg_co2e"] - metrics[component_id]["ghg_avoided_kg_co2e"]
            metrics[component_id]["acidification_net_kg_so2e"] = metrics[component_id]["acidification_caused_kg_so2e"] - metrics[component_id]["acidification_avoided_kg_so2e"]
            metrics[component_id]["eutrophication_net_kg_po4e"] = metrics[component_id]["eutrophication_caused_kg_po4e"] - metrics[component_id]["eutrophication_avoided_kg_po4e"]
            metrics[component_id]["total_energy_kwh"] = metrics[component_id]["electricity_kwh"] + metrics[component_id]["fossil_energy_kwh"] + metrics[component_id]["embodied_energy_kwh"]
            metrics[component_id]["total_cost_eur"] = metrics[component_id]["operational_cost_eur"] + metrics[component_id]["capital_cost_eur"]

    @staticmethod
    def _annual_failure_rate(
        model: dict[str, Any],
        *,
        age_years: float,
        diameter_mm: float = 0.0,
        length_km: float = 1.0,
        material: str | None = None,
    ) -> float:
        """Evaluate constant, linear, exponential or Weibull deterioration models."""
        form = model.get("model", "constant")
        if "annual_failure_rate" in model:
            base = float(model["annual_failure_rate"])
        elif form in {"linear", "exponential"}:
            predictor = (
                float(model.get("intercept", 0.0))
                + float(model.get("age_coefficient", 0.0)) * age_years
                + float(model.get("diameter_coefficient", 0.0)) * diameter_mm
            )
            base = float(np.exp(predictor)) if form == "exponential" else predictor
        elif form == "weibull":
            shape = float(model.get("shape", 1.0))
            scale = max(float(model.get("scale_years", 1.0)), 1e-12)
            base = shape / scale * max(age_years / scale, 1e-12) ** (shape - 1)
        else:
            base = float(model.get("annual_failures_per_km", 0.0))
        base *= float(model.get("material_factors", {}).get(material, 1.0))
        if model.get("basis", "per_km") == "per_km":
            base *= length_km
        return max(0.0, base)

    def _apply_asset_costs_and_failures(
        self, date: pd.Timestamp, metrics: dict[str, defaultdict[str, float]]
    ) -> None:
        finance = self.project.get("finance", {})
        discount_rate = float(finance.get("discount_rate", 0.0))
        for component_id, component in self.project["components"].items():
            if not self._component_active(component, date):
                continue
            capital = float(component.get("capital_cost_eur", 0.0))
            investment = pd.Timestamp(
                component.get("investment_date", component.get("start_date", self.project["simulation"]["start"]))
            )
            inflation = float(component.get("capital_inflation_rate", finance.get("inflation_rate", 0.0)))
            lifetime = float(component.get("lifetime_years", 0.0))
            replacement_today = False
            if date == investment:
                replacement_today = True
            elif lifetime > 0 and date.year > investment.year:
                interval = max(1, int(round(lifetime)))
                replacement_today = (
                    date.month == investment.month
                    and date.day == investment.day
                    and (date.year - investment.year) % interval == 0
                )
            if replacement_today:
                escalated = capital * (1 + inflation) ** max(
                    0.0, (date - investment).days / 365.25
                )
                metrics[component_id]["capital_cost_eur"] += escalated
            maintenance = (
                float(component.get("maintenance_cost_eur_year", 0.0))
                + capital * float(component.get("maintenance_fraction_capital_year", 0.0))
            ) / 365.25
            metrics[component_id]["maintenance_cost_eur"] += maintenance
            metrics[component_id]["operational_cost_eur"] += maintenance
            if lifetime > 0 and capital:
                crf = (
                    discount_rate * (1 + discount_rate) ** lifetime
                    / ((1 + discount_rate) ** lifetime - 1)
                    if discount_rate
                    else 1.0 / lifetime
                )
                metrics[component_id]["annualized_capital_cost_eur"] += capital * crf / 365.25

            failure_model = component.get("failure_model")
            if failure_model:
                if "age_years" in component:
                    simulation_start = pd.Timestamp(self.project["simulation"]["start"])
                    age = float(component["age_years"]) + max(
                        0.0, (date - simulation_start).days / 365.25
                    )
                else:
                    age = max(0.0, (date - investment).days / 365.25)
                expected = self._annual_failure_rate(
                    failure_model,
                    age_years=age,
                    diameter_mm=float(component.get("diameter_mm", 0.0)),
                    length_km=float(component.get("length_m", 1000.0)) / 1000.0,
                    material=component.get("material"),
                ) / 365.25
                repair = expected * float(failure_model.get("repair_cost_eur_failure", 0.0))
                metrics[component_id]["expected_failures"] += expected
                metrics[component_id]["expected_outage_hours"] += expected * float(
                    failure_model.get("duration_hours_failure", 0.0)
                )
                metrics[component_id]["failure_repair_cost_eur"] += repair
                metrics[component_id]["operational_cost_eur"] += repair

        for pipeline in self.pipeline_state.values():
            failure_model = pipeline.get("failure_model")
            component_id = pipeline["component_id"]
            if failure_model:
                expected_annual = sum(
                    self._annual_failure_rate(
                        failure_model,
                        age_years=float(cohort.get("age_years", 0.0)),
                        diameter_mm=float(pipeline.get("diameter_mm", 0.0)),
                        length_km=float(cohort.get("length_m", 0.0)) / 1000.0,
                        material=cohort.get("material", pipeline.get("material")),
                    )
                    for cohort in pipeline.get("cohorts", [])
                )
                expected = expected_annual / 365.25
                repair = expected * float(failure_model.get("repair_cost_eur_failure", 0.0))
                metrics[component_id]["expected_failures"] += expected
                metrics[component_id]["expected_outage_hours"] += expected * float(
                    failure_model.get("duration_hours_failure", 0.0)
                )
                metrics[component_id]["failure_repair_cost_eur"] += repair
                metrics[component_id]["operational_cost_eur"] += repair
            maintenance = (
                float(pipeline.get("maintenance_cost_eur_m_year", 0.0))
                * float(pipeline.get("length_m", 0.0))
                / 365.25
            )
            metrics[component_id]["maintenance_cost_eur"] += maintenance
            metrics[component_id]["operational_cost_eur"] += maintenance

    def _apply_pipeline_events(
        self, date: pd.Timestamp, metrics: dict[str, defaultdict[str, float]]
    ) -> None:
        for event in self.project.get("pipeline_events", []):
            if pd.Timestamp(event["date"]) != date:
                continue
            pipeline = self.pipeline_state[event["pipeline_id"]]
            available = sum(float(cohort["length_m"]) for cohort in pipeline["cohorts"])
            action = event.get("action", "rehabilitate")
            requested = float(event.get("length_m", available))
            if action == "add":
                length_m = requested
                self._rehabilitate_pipeline(
                    date, pipeline, event["method"], length_m, metrics,
                    "addition", remove_existing=False
                )
            elif action == "retire":
                length_m = min(requested, available)
                self._retire_pipeline(date, pipeline, length_m, metrics)
            else:
                length_m = min(requested, available)
                self._rehabilitate_pipeline(
                    date, pipeline, event["method"], length_m, metrics, "event"
                )
            if event.get("new_leakage_fraction") is not None:
                pipeline["base_leakage_fraction"] = float(
                    event["new_leakage_fraction"]
                )
                for cohort in pipeline["cohorts"]:
                    if float(cohort.get("age_years", 0.0)) == 0.0:
                        cohort["leakage_fraction"] = float(
                            event["new_leakage_fraction"]
                        )
            self._update_pipeline_performance()

    def _rehabilitate_pipeline(
        self,
        date: pd.Timestamp,
        pipeline: dict[str, Any],
        method_name: str,
        length_m: float,
        metrics: dict[str, defaultdict[str, float]],
        trigger: str,
        remove_existing: bool = True,
    ) -> None:
        if length_m <= 0:
            return
        component_id = pipeline["component_id"]
        method = self.project["rehabilitation_methods"][method_name]
        diameter_mm = float(pipeline["diameter_mm"])
        size = "small" if diameter_mm < 249 else "medium" if diameter_mm < 500 else "large"
        cost = length_m * float(method["cost_eur_m_by_size"][size])
        diesel_l = length_m * float(method["diesel_l_m_by_size"][size])
        material_name = method.get("material")
        material_mass = 0.0
        if material_name:
            thickness_m = diameter_mm / 1000.0 * float(
                method.get("thickness_ratio", 0.09)
            )
            outer_radius = diameter_mm / 2000.0
            inner_radius = max(0.0, outer_radius - thickness_m)
            volume_m3 = np.pi * (outer_radius**2 - inner_radius**2) * length_m
            material = self.project["materials"][material_name]
            material_mass = volume_m3 * float(material["density_kg_m3"])
            metrics[component_id]["embodied_energy_kwh"] += material_mass * float(
                material.get("embodied_energy_kwh_kg", 0.0)
            )
            for metric, field in (
                ("ghg_caused_kg_co2e", "ghg_kg_co2e_kg"),
                ("acidification_caused_kg_so2e", "acid_kg_so2e_kg"),
                ("eutrophication_caused_kg_po4e", "eutro_kg_po4e_kg"),
            ):
                metrics[component_id][metric] += material_mass * float(
                    material.get(field, 0.0)
                )
            metrics[component_id]["embodied_ghg_kg_co2e"] += material_mass * float(
                material.get("ghg_kg_co2e_kg", 0.0)
            )
            metrics[component_id][
                "embodied_acidification_kg_so2e"
            ] += material_mass * float(material.get("acid_kg_so2e_kg", 0.0))
            metrics[component_id][
                "embodied_eutrophication_kg_po4e"
            ] += material_mass * float(material.get("eutro_kg_po4e_kg", 0.0))
            metrics[component_id][f"material_{material_name}_kg"] += material_mass
        diesel = self.project.get("energy_sources", {}).get("diesel", {})
        metrics[component_id]["fossil_energy_kwh"] += diesel_l * float(
            diesel.get("energy_kwh_unit", 0.0)
        )
        metrics[component_id]["ghg_caused_kg_co2e"] += diesel_l * float(
            diesel.get("ghg_kg_co2e_unit", 0.0)
        )
        metrics[component_id]["fossil_ghg_kg_co2e"] += diesel_l * float(
            diesel.get("ghg_kg_co2e_unit", 0.0)
        )
        diesel_acid = diesel_l * float(diesel.get("acid_kg_so2e_unit", 0.0))
        diesel_eutro = diesel_l * float(diesel.get("eutro_kg_po4e_unit", 0.0))
        metrics[component_id]["acidification_caused_kg_so2e"] += diesel_acid
        metrics[component_id]["fossil_acidification_kg_so2e"] += diesel_acid
        metrics[component_id]["eutrophication_caused_kg_po4e"] += diesel_eutro
        metrics[component_id]["fossil_eutrophication_kg_po4e"] += diesel_eutro
        metrics[component_id]["capital_cost_eur"] += cost

        cohorts = sorted(
            pipeline["cohorts"],
            key=lambda cohort: float(cohort.get("age_years", 0.0)),
            reverse=True,
        )
        if remove_existing:
            remaining = length_m
            for cohort in cohorts:
                take = min(remaining, float(cohort["length_m"]))
                cohort["length_m"] = float(cohort["length_m"]) - take
                remaining -= take
                if remaining <= 1e-12:
                    break
        pipeline["cohorts"] = [
            cohort for cohort in cohorts if float(cohort["length_m"]) > 1e-12
        ]
        pipeline["cohorts"].append({
            "length_m": length_m,
            "age_years": 0.0,
            "material": material_name or pipeline.get("material"),
        })
        pipeline["length_m"] = sum(
            float(cohort["length_m"]) for cohort in pipeline["cohorts"]
        )
        self._material_rows.append({
            "date": date,
            "pipeline_id": pipeline["id"],
            "component_id": component_id,
            "method": method_name,
            "trigger": trigger,
            "length_m": length_m,
            "material": material_name,
            "material_mass_kg": material_mass,
            "diesel_l": diesel_l,
            "capital_cost_eur": cost,
        })

    def _retire_pipeline(
        self,
        date: pd.Timestamp,
        pipeline: dict[str, Any],
        length_m: float,
        metrics: dict[str, defaultdict[str, float]],
    ) -> None:
        component_id = pipeline["component_id"]
        remaining = length_m
        retired_mass = 0.0
        cohorts = sorted(
            pipeline["cohorts"],
            key=lambda cohort: float(cohort.get("age_years", 0.0)),
            reverse=True,
        )
        diameter_mm = float(pipeline["diameter_mm"])
        thickness_m = diameter_mm / 1000.0 * float(
            pipeline.get("thickness_ratio", 0.09)
        )
        outer_radius = diameter_mm / 2000.0
        inner_radius = max(0.0, outer_radius - thickness_m)
        area_m2 = np.pi * (outer_radius**2 - inner_radius**2)
        for cohort in cohorts:
            take = min(remaining, float(cohort["length_m"]))
            material_name = cohort.get("material")
            material = self.project.get("materials", {}).get(material_name, {})
            retired_mass += take * area_m2 * float(material.get("density_kg_m3", 0.0))
            cohort["length_m"] = float(cohort["length_m"]) - take
            remaining -= take
            if remaining <= 1e-12:
                break
        pipeline["cohorts"] = [
            cohort for cohort in cohorts if float(cohort["length_m"]) > 1e-12
        ]
        pipeline["length_m"] = sum(
            float(cohort["length_m"]) for cohort in pipeline["cohorts"]
        )
        retirement_cost = length_m * float(
            pipeline.get("retirement_cost_eur_m", 0.0)
        )
        metrics[component_id]["capital_cost_eur"] += retirement_cost
        metrics[component_id]["retired_material_kg"] += retired_mass
        self._material_rows.append({
            "date": date,
            "pipeline_id": pipeline["id"],
            "component_id": component_id,
            "method": "retirement",
            "trigger": "retirement",
            "length_m": length_m,
            "material": None,
            "material_mass_kg": -retired_mass,
            "diesel_l": 0.0,
            "capital_cost_eur": retirement_cost,
        })

    def _save_day(
        self,
        date: pd.Timestamp,
        area_state: dict[str, dict[str, Any]],
        metrics: dict[str, defaultdict[str, float]],
    ) -> None:
        finance = self.project.get("finance", {})
        base_date = pd.Timestamp(finance.get("base_date", self.project["simulation"]["start"]))
        rate = float(finance.get("discount_rate", 0.0))
        discount_factor = (1.0 + rate) ** max(0.0, (date - base_date).days / 365.25)
        for area_id, state in area_state.items():
            demand_total = sum(state["demand"].values())
            potable_total = sum(state["potable_delivered"].values())
            reuse_total = sum(state["reuse_delivered"].values())
            unmet_total = sum(state["unmet"].values())
            row: dict[str, Any] = {
                "date": date,
                "area_id": area_id,
                "population": state["population"],
                "water_demand_ml": demand_total,
                "potable_delivered_ml": potable_total,
                "reuse_delivered_ml": reuse_total,
                "delivered_total_ml": potable_total + reuse_total,
                "unmet_ml": unmet_total,
                "delivered_percent": 100 * (potable_total + reuse_total) / demand_total if demand_total else 100.0,
                "runoff_ml": state["runoff_ml"],
                "runoff_to_sewer_ml": state["runoff_remaining_ml"],
                "sanitary_sewage_ml": state.get("sanitary_ml", 0.0),
                "aquifer_recharge_ml": state["aquifer_recharge_ml"],
                "appliance_electricity_kwh": state["appliance_electricity_kwh"],
                "appliance_operational_cost_eur": state["appliance_operational_cost_eur"],
                "appliance_capital_cost_eur": state["appliance_capital_cost_eur"],
                "appliance_annualized_capital_cost_eur": state[
                    "appliance_annualized_capital_cost_eur"
                ],
            }
            for category, value in state["demand"].items():
                row[f"demand_{category}_ml"] = value
                row[f"potable_{category}_ml"] = state["potable_delivered"].get(category, 0.0)
                row[f"unmet_{category}_ml"] = state["unmet"].get(category, 0.0)
            for (reuse_type, category), value in state["reuse_delivered"].items():
                row[f"{reuse_type}_{category}_ml"] = value
            self._area_rows.append(row)
            for indoor_id, indoor_demands in state.get("indoor_demands", {}).items():
                indoor_demand = sum(indoor_demands.values())
                indoor_potable = 0.0
                indoor_reuse = 0.0
                indoor_unmet = 0.0
                for category, value in indoor_demands.items():
                    share = value / max(state["demand"].get(category, 0.0), 1e-12)
                    indoor_potable += state["potable_delivered"].get(category, 0.0) * share
                    indoor_reuse += sum(
                        amount * share
                        for (reuse_type, name), amount in state["reuse_delivered"].items()
                        if name == category
                    )
                    indoor_unmet += state["unmet"].get(category, 0.0) * share
                self._indoor_rows.append({
                    "date": date,
                    "indoor_id": indoor_id,
                    "local_area": area_id,
                    "population": state["indoor_populations"][indoor_id],
                    "water_demand_ml": indoor_demand,
                    "potable_delivered_ml": indoor_potable,
                    "reuse_delivered_ml": indoor_reuse,
                    "delivered_total_ml": indoor_potable + indoor_reuse,
                    "unmet_ml": indoor_unmet,
                })

        for component_id, values in metrics.items():
            component = self.project["components"][component_id]
            values["discounted_total_cost_eur"] = values["total_cost_eur"] / discount_factor
            self._component_rows.append(
                {"date": date, "component_id": component_id, "kind": component["kind"], **dict(values)}
            )
        system = _zero_component_metrics()
        for values in metrics.values():
            for key, value in values.items():
                system[key] += value
        area_rows_today = self._area_rows[-len(area_state) :]
        system.update(
            {
                "date": date,
                "water_demand_ml": sum(row["water_demand_ml"] for row in area_rows_today),
                "delivered_total_ml": sum(row["delivered_total_ml"] for row in area_rows_today),
                "potable_delivered_ml": sum(row["potable_delivered_ml"] for row in area_rows_today),
                "reuse_delivered_ml": sum(row["reuse_delivered_ml"] for row in area_rows_today),
                "unmet_demand_ml": sum(row["unmet_ml"] for row in area_rows_today),
                "population": sum(row["population"] for row in area_rows_today),
                "aquifer_recharge_ml": sum(row["aquifer_recharge_ml"] for row in area_rows_today),
                "electricity_kwh": system["electricity_kwh"]
                + sum(row["appliance_electricity_kwh"] for row in area_rows_today),
                "operational_cost_eur": system["operational_cost_eur"]
                + sum(row["appliance_operational_cost_eur"] for row in area_rows_today),
                "capital_cost_eur": system["capital_cost_eur"]
                + sum(row["appliance_capital_cost_eur"] for row in area_rows_today),
                "annualized_capital_cost_eur": system["annualized_capital_cost_eur"]
                + sum(
                    row["appliance_annualized_capital_cost_eur"]
                    for row in area_rows_today
                ),
            }
        )
        electricity_factor = self.project.get("energy_sources", {}).get("electricity", {})
        appliance_kwh = sum(row["appliance_electricity_kwh"] for row in area_rows_today)
        system["ghg_caused_kg_co2e"] += appliance_kwh * float(
            electricity_factor.get("ghg_kg_co2e_unit", 0.0)
        )
        system["operational_cost_eur"] += appliance_kwh * float(
            electricity_factor.get("cost_eur_unit", 0.0)
        )
        system["total_cost_eur"] = system["operational_cost_eur"] + system["capital_cost_eur"]
        system["discounted_total_cost_eur"] = system["total_cost_eur"] / discount_factor
        system["ghg_net_kg_co2e"] = (
            system["ghg_caused_kg_co2e"] - system["ghg_avoided_kg_co2e"]
        )
        distribution_metrics = [
            values
            for component_id, values in metrics.items()
            if self.project["components"][component_id]["kind"] == "distribution_main"
        ]
        delivered_quality_volume = sum(
            values["water_quality_weighted_ml"] for values in distribution_metrics
        )
        distribution_outflow = sum(
            values["outflow_ml"] for values in distribution_metrics
        )
        system["tap_water_quality_index"] = (
            delivered_quality_volume / distribution_outflow
            if distribution_outflow else np.nan
        )
        system["tap_quality_weighted_ml"] = delivered_quality_volume
        system["tap_quality_flow_ml"] = distribution_outflow
        quality = system["tap_water_quality_index"]
        system["tap_water_quality_class"] = (
            "excellent" if quality >= 0.8
            else "good" if quality >= 0.6
            else "acceptable" if quality >= 0.4
            else "poor"
        ) if np.isfinite(quality) else "unknown"
        system["delivered_percent"] = 100 * system["delivered_total_ml"] / system["water_demand_ml"] if system["water_demand_ml"] else 100.0
        self._system_rows.append(dict(system))

    def _aggregate_subcatchments(self) -> pd.DataFrame:
        if not self._area_rows:
            return pd.DataFrame()
        frame = pd.DataFrame(self._area_rows)
        mapping = {
            area_id: area.get("subcatchment", "SYSTEM")
            for area_id, area in self.project["local_areas"].items()
        }
        frame["subcatchment_id"] = frame["area_id"].map(mapping)
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        if "delivered_percent" in numeric:
            numeric.remove("delivered_percent")
        grouped = frame.groupby(["date", "subcatchment_id"], as_index=False)[numeric].sum()
        grouped["delivered_percent"] = np.where(
            grouped["water_demand_ml"] > 0,
            100 * grouped["delivered_total_ml"] / grouped["water_demand_ml"],
            100.0,
        )
        return grouped


def run_full_model(project: dict[str, Any], timeseries: pd.DataFrame) -> FullModelResult:
    """Public API equivalent to the original Toolkit's load/run/retrieve workflow."""
    return FullWaterMet2Model(project, timeseries).run()
