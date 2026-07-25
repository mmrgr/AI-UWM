from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .full_engine import FullWaterMet2Model, FullModelResult


def performance_statistics(observed: pd.Series, simulated: pd.Series) -> dict[str, float]:
    """NSE, RSR and PBIAS used in the paper's calibration/validation."""
    aligned = pd.concat([observed.rename("observed"), simulated.rename("simulated")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("观测与模拟序列没有重叠的有效值")
    obs = aligned["observed"].to_numpy(dtype=float)
    sim = aligned["simulated"].to_numpy(dtype=float)
    denominator = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((obs - sim) ** 2) / denominator if denominator else np.nan
    rmse = float(np.sqrt(np.mean((obs - sim) ** 2)))
    std = float(np.std(obs, ddof=0))
    rsr = rmse / std if std else np.nan
    pbias = 100.0 * np.sum(obs - sim) / np.sum(obs) if np.sum(obs) else np.nan
    return {"nse": float(nse), "rsr": float(rsr), "pbias_percent": float(pbias), "rmse": rmse}


def extract_series(result: FullModelResult, table: str, column: str, selector: str | None = None) -> pd.Series:
    frame = getattr(result, table)
    if selector:
        key, value = selector.split("=", 1)
        frame = frame[frame[key].astype(str) == value]
    if column not in frame:
        raise KeyError(f"结果表 {table} 不含列 {column}")
    return frame.set_index("date")[column]


def _set_parameter(project: dict[str, Any], path: str, value: Any) -> None:
    target: Any = project
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


@dataclass
class CalibrationResult:
    trials: pd.DataFrame
    best_project: dict[str, Any]


def grid_calibrate(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    observed: pd.Series,
    parameters: dict[str, list[float]],
    result_table: str,
    result_column: str,
    selector: str | None = None,
    objective: str = "nse",
    frequency: str | None = None,
    aggregation: str = "sum",
) -> CalibrationResult:
    """Transparent exhaustive calibration for small strategic-model parameter grids."""
    paths = list(parameters)
    records: list[dict[str, Any]] = []
    best_score = -np.inf if objective == "nse" else np.inf
    best_project: dict[str, Any] | None = None
    for values in itertools.product(*(parameters[path] for path in paths)):
        trial = copy.deepcopy(project)
        for path, value in zip(paths, values):
            _set_parameter(trial, path, value)
        result = FullWaterMet2Model(trial, timeseries).run()
        simulated = extract_series(result, result_table, result_column, selector)
        trial_observed = observed
        if frequency:
            if aggregation not in {"sum", "mean", "last"}:
                raise ValueError("aggregation must be sum, mean, or last")
            simulated = getattr(simulated.resample(frequency), aggregation)()
            trial_observed = getattr(observed.resample(frequency), aggregation)()
        stats = performance_statistics(trial_observed, simulated)
        record = {path: value for path, value in zip(paths, values)}
        record.update(stats)
        records.append(record)
        score = stats[objective]
        better = score > best_score if objective == "nse" else score < best_score
        if better:
            best_score = score
            best_project = trial
    if best_project is None:
        raise ValueError("没有完成任何校准试验")
    trials = pd.DataFrame(records).sort_values(objective, ascending=objective != "nse")
    return CalibrationResult(trials=trials, best_project=best_project)


def summarize_kpis(result: FullModelResult) -> dict[str, float]:
    daily = result.system_daily
    demand = daily["water_demand_ml"].sum()
    delivered = daily["delivered_total_ml"].sum()
    years = max(1.0, (daily["date"].max() - daily["date"].min()).days / 365.25)
    values = {
        "reliability_fraction": delivered / demand if demand else 1.0,
        "total_unmet_ml": float(daily["unmet_demand_ml"].sum()),
        "mean_annual_leakage_ml": float(daily["leakage_ml"].sum() / years),
        "mean_annual_ghg_net_kg_co2e": float(daily["ghg_net_kg_co2e"].sum() / years),
        "mean_annual_acidification_net_kg_so2e": float(daily["acidification_net_kg_so2e"].sum() / years),
        "mean_annual_eutrophication_net_kg_po4e": float(daily["eutrophication_net_kg_po4e"].sum() / years),
        "present_total_cost_eur": float(daily["discounted_total_cost_eur"].sum()),
        "total_energy_generated_kwh": float(daily["energy_generated_kwh"].sum()),
        "mean_annual_expected_failures": float(daily["expected_failures"].sum() / years),
        "mean_annual_flooded_area_m2_days": float(
            result.flood_daily.get("flooded_area_m2", pd.Series(dtype=float)).sum()
            / years
        ),
        "mean_annual_aquifer_recharge_ml": float(
            daily.get("aquifer_recharge_ml", pd.Series(dtype=float)).sum() / years
        ),
        "mean_annual_imported_water_ml": float(
            daily.get("imported_water_ml", pd.Series(dtype=float)).sum() / years
        ),
        "mean_annual_exported_water_ml": float(
            daily.get("exported_water_ml", pd.Series(dtype=float)).sum() / years
        ),
        "mean_annual_cso_ml": float(daily.get("cso_ml", pd.Series(dtype=float)).sum() / years),
        "mean_annual_untreated_wastewater_ml": float(
            daily.get("untreated_ml", pd.Series(dtype=float)).sum() / years
        ),
    }
    if not result.risk_summary.empty:
        for row in result.risk_summary.itertuples(index=False):
            values[f"risk_{row.risk_code}_probability"] = float(row.probability)
            values[f"risk_{row.risk_code}_score"] = float(row.cumulative_risk_score)
    return values


def _sample_distribution(rng: np.random.Generator, spec: dict[str, Any]) -> float:
    distribution = spec["distribution"]
    if distribution == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if distribution == "normal":
        return float(rng.normal(spec["mean"], spec["sd"]))
    if distribution == "triangular":
        return float(rng.triangular(spec["low"], spec["mode"], spec["high"]))
    if distribution == "lognormal":
        return float(rng.lognormal(spec["mean_log"], spec["sd_log"]))
    if distribution == "choice":
        return float(rng.choice(spec["values"], p=spec.get("probabilities")))
    raise ValueError(f"未知分布 {distribution}")


def monte_carlo(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    uncertain_parameters: list[dict[str, Any]],
    samples: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for sample in range(samples):
        trial = copy.deepcopy(project)
        record: dict[str, Any] = {"sample": sample}
        for spec in uncertain_parameters:
            value = _sample_distribution(rng, spec)
            _set_parameter(trial, spec["path"], value)
            record[spec["path"]] = value
        record.update(summarize_kpis(FullWaterMet2Model(trial, timeseries).run()))
        records.append(record)
    samples_frame = pd.DataFrame(records)
    kpi_columns = [column for column in samples_frame if column not in {"sample"} and column not in {spec["path"] for spec in uncertain_parameters}]
    percentiles = samples_frame[kpi_columns].quantile([0.05, 0.5, 0.95]).T
    percentiles.columns = ["p05", "p50", "p95"]
    return samples_frame, percentiles.reset_index(names="kpi")


def compromise_programming_rank(
    alternatives: pd.DataFrame,
    criteria: dict[str, dict[str, Any]],
    p: float = 2.0,
) -> pd.DataFrame:
    """Rank strategies using the Compromise Programming method used in the Oslo report."""
    scored = alternatives.copy()
    distance_terms = []
    for column, spec in criteria.items():
        values = scored[column].astype(float)
        minimum, maximum = values.min(), values.max()
        if maximum == minimum:
            normalized = pd.Series(0.0, index=values.index)
        elif spec.get("goal", "min") == "min":
            normalized = (values - minimum) / (maximum - minimum)
        else:
            normalized = (maximum - values) / (maximum - minimum)
        weight = float(spec.get("weight", 1.0))
        scored[f"normalized_{column}"] = normalized
        distance_terms.append(weight * normalized**p)
    scored["compromise_distance"] = sum(distance_terms) ** (1.0 / p)
    scored["rank"] = scored["compromise_distance"].rank(method="min").astype(int)
    return scored.sort_values(["rank", "compromise_distance"])


def pareto_grid_optimize(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    decisions: dict[str, list[Any]],
    objectives: dict[str, str],
) -> pd.DataFrame:
    """Enumerate discrete interventions and return all trials with Pareto membership."""
    paths = list(decisions)
    records: list[dict[str, Any]] = []
    for values in itertools.product(*(decisions[path] for path in paths)):
        trial = copy.deepcopy(project)
        for path, value in zip(paths, values):
            _set_parameter(trial, path, value)
        record = {path: value for path, value in zip(paths, values)}
        record.update(summarize_kpis(FullWaterMet2Model(trial, timeseries).run()))
        records.append(record)
    frame = pd.DataFrame(records)
    pareto = np.ones(len(frame), dtype=bool)
    objective_values = frame[list(objectives)].to_numpy(dtype=float)
    signs = np.array([1.0 if objectives[name] == "min" else -1.0 for name in objectives])
    minimized = objective_values * signs
    for index, candidate in enumerate(minimized):
        dominated = np.all(minimized <= candidate, axis=1) & np.any(
            minimized < candidate, axis=1
        )
        if dominated.any():
            pareto[index] = False
    frame["is_pareto"] = pareto
    return frame.sort_values("is_pareto", ascending=False).reset_index(drop=True)
