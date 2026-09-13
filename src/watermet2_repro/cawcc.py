"""Research-facing helpers for the CAWCC framework in the opening proposal.

The engine remains a daily WaterMet2 balance model.  This module adds the
explicit scenario labels, constraint specification and transparent intraday
peak proxy used to compare annual/daily results with an hourly stress case.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


REPORT_SYSTEM_STATES: dict[str, str] = {
    "S0": "现状基准",
    "S1": "AI节点接入",
    "S2": "扩容与高温压力",
    "S3": "回用/节水干预后",
}
REPORT_PRESSURES: dict[str, str] = {
    "G0": "常态气候",
    "G1": "高温",
    "G2": "干旱供给压力",
    "G3": "高温-干旱复合极端",
}


def report_constraint_spec(
    project: dict[str, Any],
    *,
    reliability: float = 0.99,
    wtw_utilization: float = 1.0,
    wwtw_utilization: float = 1.0,
    reuse_utilization: float = 1.0,
) -> dict[str, dict[str, float | str]]:
    """Build the Table-1 CAWCC constraints from the supplied project.

    Local connection and grid constraints are included only when the AI
    component declares their capacities; this keeps legacy projects valid.
    """
    constraints: dict[str, dict[str, float | str]] = {
        "system_reliability_fraction": {"operator": ">=", "value": float(reliability)},
        "domestic_unmet_ml": {"operator": "<=", "value": 0.0},
        "max_wtw_utilization": {"operator": "<=", "value": float(wtw_utilization)},
        "max_wwtw_utilization": {"operator": "<=", "value": float(wwtw_utilization)},
        "max_reuse_utilization": {"operator": "<=", "value": float(reuse_utilization)},
    }
    data_centers = [c for c in project.get("components", {}).values() if c.get("kind") == "data_center"]
    if any(c.get("local_connection_capacity_ml_day") is not None for c in data_centers):
        constraints["max_local_connection_ratio"] = {"operator": "<=", "value": 1.0}
    if any(c.get("grid_connection_capacity_mw") is not None for c in data_centers):
        constraints["max_daily_average_grid_connection_ratio"] = {"operator": "<=", "value": 1.0}
    return constraints


def build_intraday_profile(
    *,
    training_fraction: float = 0.65,
    inference_fraction: float = 0.35,
    inference_peak_factor: float = 1.35,
    peak_hours: Iterable[int] = range(10, 18),
) -> pd.DataFrame:
    """Return a normalized 24-hour AI load profile for the hourly proxy.

    Training and inference fractions are recorded explicitly so the profile
    can be reported and replaced by measured telemetry later.
    """
    if training_fraction < 0 or inference_fraction < 0 or training_fraction + inference_fraction <= 0:
        raise ValueError("training_fraction and inference_fraction must be non-negative and non-zero")
    if inference_peak_factor <= 0:
        raise ValueError("inference_peak_factor must be positive")
    hours = np.arange(24)
    peak_set = set(int(hour) for hour in peak_hours)
    inference = np.where(np.isin(hours, list(peak_set)), float(inference_peak_factor), 1.0)
    training = np.ones(24, dtype=float)
    raw = training_fraction * training + inference_fraction * inference
    profile = raw / raw.mean()
    return pd.DataFrame({"hour": hours, "load_multiplier": profile})


def intraday_peak_proxy(
    daily: pd.DataFrame,
    *,
    date_column: str = "date",
    flow_columns: Iterable[str] = ("external_withdrawal_ml", "consumption_ml", "return_flow_ml"),
    profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand daily AI totals to a mass-preserving 24-hour stress proxy.

    The proxy is not a substitute for telemetry; it makes the proposal's
    hourly/peak boundary auditable while preserving each daily total.
    """
    if date_column not in daily:
        raise ValueError(f"daily data must contain {date_column}")
    profile = profile.copy() if profile is not None else build_intraday_profile()
    if list(profile.columns) != ["hour", "load_multiplier"] or len(profile) != 24:
        raise ValueError("profile must contain 24 rows with hour and load_multiplier columns")
    if (profile["load_multiplier"] < 0).any() or profile["load_multiplier"].sum() <= 0:
        raise ValueError("profile load multipliers must be non-negative and non-zero")
    multipliers = profile["load_multiplier"].to_numpy(dtype=float)
    multipliers = multipliers / multipliers.sum()
    rows: list[dict[str, Any]] = []
    columns = [column for column in flow_columns if column in daily]
    for _, source in daily.iterrows():
        timestamp = pd.Timestamp(source[date_column]).normalize()
        for hour, multiplier in enumerate(multipliers):
            record: dict[str, Any] = {"date": timestamp + pd.Timedelta(hours=hour), "hour": hour}
            for column in columns:
                record[column] = float(source[column]) * float(multiplier)
            rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if "external_withdrawal_ml" in result:
        daily_max = result.groupby(result["date"].dt.normalize())["external_withdrawal_ml"].transform("max")
        result["peak_hour"] = np.isclose(result["external_withdrawal_ml"], daily_max)
    else:
        result["peak_hour"] = False
    return result
