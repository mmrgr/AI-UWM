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
    definitions = constraints.get("constraints", constraints)
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
    classified = capacity_scan.copy()
    classified["constraint_status"] = ["safe" if item[0] else "high_risk" for item in evaluations]
    if first_failed_index is not None and first_failed_index > 0:
        classified.loc[first_failed_index, "constraint_status"] = "stress"
    return {
        "maximum_safe_ai_capacity_mw": safe_capacity,
        "first_failed_capacity_mw": first_failed,
        "limiting_constraint": limiting,
        "constraint_margin": margin,
        "threshold_interval_mw": [safe_capacity, first_failed],
        "capacity_scan": classified,
    }
