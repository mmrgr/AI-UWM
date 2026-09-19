from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .ai_capacity import find_ai_carrying_capacity, scan_ai_capacity


def morris_sensitivity(
    evaluator: Callable[[dict[str, float]], float],
    bounds: dict[str, tuple[float, float]],
    trajectories: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute Morris elementary-effect μ, μ* and σ without external dependencies."""
    if trajectories < 2:
        raise ValueError("Morris analysis requires at least two trajectories")
    if not bounds or any(low > high for low, high in bounds.values()):
        raise ValueError("Sensitivity bounds require low <= high")
    rng = np.random.default_rng(seed)
    effects = {name: [] for name in bounds}
    for _ in range(trajectories):
        base = {name: float(rng.uniform(low, high)) for name, (low, high) in bounds.items()}
        base_value = float(evaluator(base))
        for name, (low, high) in bounds.items():
            delta = 0.1 * (high - low)
            if delta == 0:
                effects[name].append(0.0)
                continue
            changed = dict(base)
            changed[name] = min(high, base[name] + delta) if base[name] <= high - delta else max(low, base[name] - delta)
            actual_delta = changed[name] - base[name]
            effects[name].append((float(evaluator(changed)) - base_value) / actual_delta)
    return pd.DataFrame([
        {"parameter": name, "mu": float(np.mean(values)), "mu_star": float(np.mean(np.abs(values))), "sigma": float(np.std(values, ddof=1))}
        for name, values in effects.items()
    ]).sort_values("mu_star", ascending=False).reset_index(drop=True)


def sobol_sensitivity(
    evaluator: Callable[[dict[str, float]], float],
    bounds: dict[str, tuple[float, float]],
    samples: int = 256,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate first-order and total-order Sobol indices with Saltelli matrices."""
    if samples < 16:
        raise ValueError("Sobol analysis requires at least 16 base samples")
    if not bounds or any(low > high for low, high in bounds.values()):
        raise ValueError("Sensitivity bounds require low <= high")
    names = list(bounds)
    rng = np.random.default_rng(seed)
    a = rng.random((samples, len(names)))
    b = rng.random((samples, len(names)))
    low = np.array([bounds[name][0] for name in names])
    span = np.array([bounds[name][1] - bounds[name][0] for name in names])
    a, b = low + a * span, low + b * span
    evaluate = lambda row: float(evaluator(dict(zip(names, row))))
    ya = np.array([evaluate(row) for row in a])
    yb = np.array([evaluate(row) for row in b])
    variance = float(np.var(np.concatenate((ya, yb)), ddof=1))
    records = []
    for index, name in enumerate(names):
        ab = a.copy()
        ab[:, index] = b[:, index]
        yab = np.array([evaluate(row) for row in ab])
        first = float(np.mean(yb * (yab - ya)) / variance) if variance else 0.0
        total = float(0.5 * np.mean((ya - yab) ** 2) / variance) if variance else 0.0
        records.append({"parameter": name, "S1": first, "ST": total})
    return pd.DataFrame(records).sort_values("ST", ascending=False).reset_index(drop=True)


def _set_parameter(project: dict[str, Any], path: str, value: float) -> None:
    target: Any = project
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


def probabilistic_ai_capacity_threshold(
    project: dict[str, Any],
    timeseries: pd.DataFrame,
    parameter_ranges: dict[str, tuple[float, float]],
    capacities_mw: list[float],
    constraints: dict[str, Any],
    samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not capacities_mw:
        raise ValueError("capacities_mw must not be empty")
    if any(low > high for low, high in parameter_ranges.values()):
        raise ValueError("parameter ranges require low <= high")
    rng = np.random.default_rng(seed)
    rows = []
    for sample_id in range(samples):
        trial = copy.deepcopy(project)
        sampled = {}
        for path, (low, high) in parameter_ranges.items():
            sampled[path] = float(rng.uniform(low, high))
            _set_parameter(trial, path, sampled[path])
        threshold = find_ai_carrying_capacity(
            scan_ai_capacity(trial, timeseries, capacities_mw=capacities_mw), constraints
        )
        rows.append({"sample_id": sample_id, **sampled, **{key: value for key, value in threshold.items() if key != "capacity_scan"}})
    return pd.DataFrame(rows)


def capacity_exceedance_probability(
    threshold_samples: pd.DataFrame, proposed_capacity_mw: float
) -> float:
    if "maximum_safe_ai_capacity_mw" not in threshold_samples:
        raise KeyError("threshold_samples must contain maximum_safe_ai_capacity_mw")
    proposed = float(proposed_capacity_mw)
    if not np.isfinite(proposed):
        raise ValueError("proposed_capacity_mw must be finite")
    thresholds = pd.to_numeric(
        threshold_samples["maximum_safe_ai_capacity_mw"], errors="coerce"
    ).dropna()
    if thresholds.empty:
        return float("nan")
    return float((thresholds < proposed).mean())
