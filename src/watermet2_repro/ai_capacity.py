from __future__ import annotations

import copy
import operator
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .ai_metrics import summarize_ai_water_kpis
from .full_engine import FullWaterMet2Model


def _data_center_ids(project: dict[str, Any], data_center_id: str | None) -> list[str]:
    ids = [key for key, value in project["components"].items() if value.get("kind") == "data_center"]
    if data_center_id:
        if data_center_id not in ids:
            raise KeyError(f"Unknown data center: {data_center_id}")
        return [data_center_id]
    if not ids:
        raise ValueError("Project contains no data_center component")
    return ids


def scan_ai_capacity(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    min_mw: float = 0.0,
    max_mw: float = 2000.0,
    step_mw: float = 25.0,
    *,
    capacities_mw: Iterable[float] | None = None,
    data_center_id: str | None = None,
) -> pd.DataFrame:
    if step_mw <= 0 or max_mw < min_mw:
        raise ValueError("Capacity range requires step_mw > 0 and max_mw >= min_mw")
    capacities = list(capacities_mw) if capacities_mw is not None else list(np.arange(min_mw, max_mw + step_mw * 0.5, step_mw))
    ids = _data_center_ids(project, data_center_id)
    records: list[dict[str, float]] = []
    for capacity in capacities:
        trial = copy.deepcopy(project)
        for component_id in ids:
            trial["components"][component_id]["installed_it_capacity_mw"] = float(capacity)
            trial["components"][component_id].pop("capacity_schedule", None)
        result = FullWaterMet2Model(trial, timeseries).run()
        records.append({"ai_capacity_mw": float(capacity), **summarize_ai_water_kpis(result, trial)})
    return pd.DataFrame(records).sort_values("ai_capacity_mw").reset_index(drop=True)


_OPERATORS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq}


def find_ai_carrying_capacity(
    capacity_scan: pd.DataFrame,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    if capacity_scan.empty or "ai_capacity_mw" not in capacity_scan:
        raise ValueError("capacity_scan must contain ai_capacity_mw rows")
    capacity_scan = capacity_scan.sort_values("ai_capacity_mw").reset_index(drop=True)
    if capacity_scan["ai_capacity_mw"].duplicated().any():
        raise ValueError("ai_capacity_mw values must be unique")
    definitions = constraints.get("constraints", constraints)
    if not definitions:
        raise ValueError("at least one carrying-capacity constraint is required")
    evaluations: list[tuple[bool, str | None, float]] = []
    for row in capacity_scan.itertuples(index=False):
        failed: list[tuple[str, float]] = []
        for metric, definition in definitions.items():
            operation = definition["operator"]
            if operation not in _OPERATORS or metric not in capacity_scan:
                raise ValueError(f"Invalid carrying-capacity constraint: {metric} {operation}")
            actual = float(getattr(row, metric))
            target = float(definition["value"])
            if not _OPERATORS[operation](actual, target):
                failed.append((metric, actual - target if operation.startswith("<") else target - actual))
        evaluations.append((not failed, failed[0][0] if failed else None, failed[0][1] if failed else 0.0))
    safe_indices = [index for index, value in enumerate(evaluations) if value[0]]
    failed_indices = [index for index, value in enumerate(evaluations) if not value[0]]
    safe_capacity = float(capacity_scan.iloc[max(safe_indices)]["ai_capacity_mw"]) if safe_indices else None
    first_failed_index = min(failed_indices) if failed_indices else None
    first_failed = float(capacity_scan.iloc[first_failed_index]["ai_capacity_mw"]) if first_failed_index is not None else None
    limiting = evaluations[first_failed_index][1] if first_failed_index is not None else None
    margin = evaluations[first_failed_index][2] if first_failed_index is not None else None
    # A constrained system can re-enter feasibility after an intervention or
    # source-switch threshold.  Keep the complete feasible set instead of
    # incorrectly treating the largest safe point as the first boundary.
    intervals: list[list[float]] = []
    for index in safe_indices:
        capacity = float(capacity_scan.iloc[index]["ai_capacity_mw"])
        if not intervals or index - 1 not in safe_indices:
            intervals.append([capacity, capacity])
        else:
            intervals[-1][1] = capacity
    first_safe_before_failure = (
        float(capacity_scan.iloc[first_failed_index - 1]["ai_capacity_mw"])
        if first_failed_index is not None and first_failed_index > 0
        and evaluations[first_failed_index - 1][0]
        else None
    )
    classified = capacity_scan.copy()
    classified["constraint_status"] = ["safe" if item[0] else "high_risk" for item in evaluations]
    if first_failed_index is not None and first_failed_index > 0:
        classified.loc[first_failed_index, "constraint_status"] = "stress"
    threshold_lower = (
        first_safe_before_failure
        if first_failed_index is not None
        else safe_capacity
    )
    return {
        "maximum_safe_ai_capacity_mw": safe_capacity,
        "first_failed_capacity_mw": first_failed,
        "limiting_constraint": limiting,
        "constraint_margin": margin,
        "threshold_interval_mw": [threshold_lower, first_failed],
        "feasible_intervals_mw": intervals,
        "reentrant_feasibility": bool(
            first_failed_index is not None and any(index > first_failed_index for index in safe_indices)
        ),
        "capacity_scan": classified,
    }


def summarize_constraint_boundaries(
    capacity_scan: pd.DataFrame,
    constraints: dict[str, Any],
) -> pd.DataFrame:
    """Report the first failing capacity and margin for every CAWCC constraint.

    This table is the bottleneck-migration input: unlike a single limiting
    constraint it retains constraints that become active later or re-enter
    feasibility after an intervention.
    """
    if capacity_scan.empty or "ai_capacity_mw" not in capacity_scan:
        raise ValueError("capacity_scan must contain ai_capacity_mw rows")
    definitions = constraints.get("constraints", constraints)
    rows: list[dict[str, Any]] = []
    ordered = capacity_scan.sort_values("ai_capacity_mw").reset_index(drop=True)
    for metric, definition in definitions.items():
        operation = definition.get("operator")
        if operation not in _OPERATORS or metric not in ordered:
            raise ValueError(f"Invalid carrying-capacity constraint: {metric} {operation}")
        target = float(definition["value"])
        passed = ordered[metric].astype(float).map(lambda value: _OPERATORS[operation](value, target))
        failures = ordered.index[~passed]
        first = int(failures[0]) if len(failures) else None
        rows.append({
            "constraint": metric,
            "operator": operation,
            "target": target,
            "first_failed_capacity_mw": float(ordered.loc[first, "ai_capacity_mw"]) if first is not None else None,
            "maximum_safe_capacity_mw": float(ordered.loc[passed[passed].index.max(), "ai_capacity_mw"]) if passed.any() else None,
            "first_failure_margin": (
                float(ordered.loc[first, metric]) - target
                if first is not None and operation.startswith("<")
                else target - float(ordered.loc[first, metric])
                if first is not None else None
            ),
            "reentrant_feasibility": bool(first is not None and passed.iloc[first + 1 :].any()),
        })
    return pd.DataFrame(rows).sort_values(
        ["first_failed_capacity_mw", "constraint"], na_position="last"
    ).reset_index(drop=True)


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    target: Any = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


def evaluate_ai_interventions(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    interventions: Iterable[dict[str, Any]],
    constraints: dict[str, Any],
    *,
    capacities_mw: Iterable[float],
) -> pd.DataFrame:
    """Re-scan CAWCC after each one-at-a-time capacity intervention."""
    records: list[dict[str, Any]] = []
    for intervention in interventions:
        trial = copy.deepcopy(project)
        for change in intervention.get("set", []):
            _set_path(trial, str(change["path"]), change["value"])
        scan = scan_ai_capacity(trial, timeseries, capacities_mw=capacities_mw)
        threshold = find_ai_carrying_capacity(scan, constraints)
        records.append({
            "intervention": intervention.get("name", "unnamed"),
            "maximum_safe_ai_capacity_mw": threshold["maximum_safe_ai_capacity_mw"],
            "first_failed_capacity_mw": threshold["first_failed_capacity_mw"],
            "limiting_constraint": threshold["limiting_constraint"],
        })
    return pd.DataFrame(records)
