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
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


@dataclass
class CalibrationResult:
    trials: pd.DataFrame
    best_project: dict[str, Any]


@dataclass(frozen=True)
class AHPResult:
    labels: tuple[str, ...]
    weights: np.ndarray
    principal_eigenvalue: float
    consistency_index: float
    consistency_ratio: float
    is_consistent: bool


_AHP_RANDOM_INDEX = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
    13: 1.56,
    14: 1.57,
    15: 1.59,
}


def analytic_hierarchy_weights(
    pairwise: np.ndarray | list[list[float]],
    labels: list[str] | tuple[str, ...] | None = None,
    consistency_threshold: float = 0.10,
) -> AHPResult:
    """Calculate AHP priorities and Saaty's consistency ratio."""
    matrix = np.asarray(pairwise, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("AHP pairwise matrix must be non-empty and square")
    size = matrix.shape[0]
    names = tuple(labels or [str(index + 1) for index in range(size)])
    if len(names) != size or len(set(names)) != size:
        raise ValueError("AHP labels must be unique and match the matrix size")
    if not np.isfinite(matrix).all() or (matrix <= 0).any():
        raise ValueError("AHP pairwise values must be finite and positive")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
        raise ValueError("AHP pairwise matrix diagonal must equal 1")
    if not np.allclose(matrix * matrix.T, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("AHP pairwise matrix must be reciprocal")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    principal_index = int(np.argmax(eigenvalues.real))
    principal_eigenvalue = float(eigenvalues[principal_index].real)
    weights = np.abs(eigenvectors[:, principal_index].real)
    weights /= weights.sum()
    consistency_index = (
        max(0.0, (principal_eigenvalue - size) / (size - 1)) if size > 2 else 0.0
    )
    random_index = _AHP_RANDOM_INDEX.get(size)
    if random_index is None:
        raise ValueError("AHP consistency ratio supports at most 15 criteria")
    consistency_ratio = consistency_index / random_index if random_index else 0.0
    if abs(consistency_ratio) < 1e-12:
        consistency_ratio = 0.0
    return AHPResult(
        labels=names,
        weights=weights,
        principal_eigenvalue=principal_eigenvalue,
        consistency_index=consistency_index,
        consistency_ratio=consistency_ratio,
        is_consistent=consistency_ratio <= consistency_threshold,
    )


def analytic_hierarchy_rank(
    alternatives: pd.DataFrame,
    criteria: dict[str, dict[str, Any]],
    criteria_pairwise: np.ndarray | list[list[float]],
    alternative_pairwise: dict[str, np.ndarray | list[list[float]]] | None = None,
    consistency_threshold: float = 0.10,
) -> tuple[pd.DataFrame, AHPResult]:
    """Rank alternatives with AHP criteria weights and optional pairwise alternatives."""
    names = list(criteria)
    missing = [name for name in names if name not in alternatives]
    if missing:
        raise KeyError(f"Alternatives table is missing criteria: {missing}")
    if alternatives.empty:
        raise ValueError("Alternatives table is empty")
    diagnostics = analytic_hierarchy_weights(
        criteria_pairwise,
        labels=names,
        consistency_threshold=consistency_threshold,
    )
    scored = alternatives.copy()
    local_priorities: list[np.ndarray] = []
    pairwise_by_criterion = alternative_pairwise or {}
    for name in names:
        if name in pairwise_by_criterion:
            result = analytic_hierarchy_weights(
                pairwise_by_criterion[name],
                labels=[str(index) for index in alternatives.index],
                consistency_threshold=consistency_threshold,
            )
            local = result.weights
            scored[f"ahp_local_{name}"] = local
            scored[f"ahp_consistency_ratio_{name}"] = result.consistency_ratio
        else:
            values = scored[name].astype(float)
            minimum, maximum = float(values.min()), float(values.max())
            if np.isclose(maximum, minimum, rtol=1e-9, atol=1e-12):
                local = np.full(len(values), 1.0 / len(values), dtype=float)
            elif criteria[name].get("goal", "min") == "max":
                local = ((values - minimum) / (maximum - minimum)).to_numpy()
            else:
                local = ((maximum - values) / (maximum - minimum)).to_numpy()
            local = local / local.sum()
            scored[f"ahp_local_{name}"] = local
        local_priorities.append(np.asarray(local, dtype=float))
    scored["ahp_score"] = np.vstack(local_priorities).T @ diagnostics.weights
    scored["rank"] = scored["ahp_score"].rank(method="min", ascending=False).astype(int)
    return scored.sort_values(["rank", "ahp_score"], ascending=[True, False]), diagnostics


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
        if np.isclose(maximum, minimum, rtol=1e-9, atol=1e-12):
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


@dataclass
class DecisionProblemResult:
    runs: pd.DataFrame
    decision_matrix: pd.DataFrame
    rankings: pd.DataFrame


def _apply_decision_definition(project: dict[str, Any], definition: dict[str, Any]) -> None:
    for change in definition.get("set", []):
        _set_parameter(project, change["path"], change["value"])
    if definition.get("interventions"):
        project.setdefault("interventions", []).extend(copy.deepcopy(definition["interventions"]))


def _named_definitions(
    specification: dict[str, Any], key: str, default_name: str
) -> list[dict[str, Any]]:
    definitions = specification.get(key) or [{"name": default_name}]
    names = [definition.get("name") for definition in definitions]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{key} must have unique non-empty names")
    return definitions


def evaluate_decision_problem(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    specification: dict[str, Any],
) -> DecisionProblemResult:
    """Evaluate scenario-strategy combinations and rank them for preference groups."""
    scenarios = _named_definitions(specification, "scenarios", "baseline")
    strategies = _named_definitions(specification, "strategies", "BAU")
    metrics = specification.get("metrics", {})
    if not metrics:
        raise ValueError("Decision problem must define at least one metric")

    run_records: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_project = copy.deepcopy(project)
        _apply_decision_definition(scenario_project, scenario)
        for strategy in strategies:
            trial = copy.deepcopy(scenario_project)
            _apply_decision_definition(trial, strategy)
            result = FullWaterMet2Model(trial, timeseries).run()
            record: dict[str, Any] = {
                "scenario": scenario["name"],
                "strategy": strategy["name"],
                **summarize_kpis(result),
            }
            for custom in specification.get("custom_indicators", []):
                scenario_match = custom.get("scenario", "*") in {"*", scenario["name"]}
                strategy_match = custom.get("strategy", "*") in {"*", strategy["name"]}
                if scenario_match and strategy_match:
                    record[custom["metric"]] = custom["value"]
            run_records.append(record)

    runs = pd.DataFrame(run_records)
    missing = [name for name in metrics if name not in runs or runs[name].isna().any()]
    if missing:
        raise ValueError(f"Decision metrics are missing values: {missing}")
    decision_matrix = runs[["scenario", "strategy", *metrics]].copy()

    preference_groups = specification.get("preference_groups") or {
        "equal": {"method": "cp", "weights": {name: 1.0 for name in metrics}}
    }
    ranking_frames: list[pd.DataFrame] = []
    for scenario_name, scenario_frame in decision_matrix.groupby("scenario", sort=False):
        for group_name, preference in preference_groups.items():
            method = preference.get("method", "cp").lower()
            if method == "cp":
                weights = preference.get("weights", {})
                criteria = {
                    name: {
                        **definition,
                        "weight": float(weights.get(name, definition.get("weight", 1.0))),
                    }
                    for name, definition in metrics.items()
                }
                ranked = compromise_programming_rank(
                    scenario_frame.copy(), criteria, p=float(preference.get("p", 2.0))
                )
                ranked["consistency_ratio"] = np.nan
                ranked["is_consistent"] = True
            elif method == "ahp":
                if "pairwise" not in preference:
                    raise ValueError(f"AHP preference group {group_name} is missing pairwise")
                ranked, diagnostics = analytic_hierarchy_rank(
                    scenario_frame.copy(),
                    metrics,
                    preference["pairwise"],
                    alternative_pairwise=preference.get("alternative_pairwise"),
                    consistency_threshold=float(
                        preference.get("consistency_threshold", 0.10)
                    ),
                )
                if preference.get("require_consistency", True) and not diagnostics.is_consistent:
                    raise ValueError(
                        f"AHP preference group {group_name} is inconsistent "
                        f"(CR={diagnostics.consistency_ratio:.4f})"
                    )
                if preference.get("require_consistency", True):
                    inconsistent = [
                        column.removeprefix("ahp_consistency_ratio_")
                        for column in ranked
                        if column.startswith("ahp_consistency_ratio_")
                        and float(ranked[column].iloc[0])
                        > float(preference.get("consistency_threshold", 0.10))
                    ]
                    if inconsistent:
                        raise ValueError(
                            f"AHP preference group {group_name} has inconsistent "
                            f"alternative matrices: {inconsistent}"
                        )
                ranked["consistency_ratio"] = diagnostics.consistency_ratio
                ranked["is_consistent"] = diagnostics.is_consistent
            else:
                raise ValueError(f"Unknown decision method: {method}")
            ranked["scenario"] = scenario_name
            ranked["preference_group"] = group_name
            ranked["method"] = method
            ranking_frames.append(ranked)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    return DecisionProblemResult(runs=runs, decision_matrix=decision_matrix, rankings=rankings)
