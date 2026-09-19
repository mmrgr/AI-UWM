"""R2 uncertainty block: Morris screening, Sobol indices, probabilistic CAWCC.

Kept separate from `run_r2.py` because it is by far the most expensive block
(Sobol needs N x (2D+2) full-engine calls) and therefore needs to be re-runnable
on its own whenever the sensitivity parameter set changes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from aiuwm.ai_metrics import summarize_ai_water_kpis
from aiuwm.full_engine import FullAIUWMModel
from aiuwm.sensitivity import (
    capacity_exceedance_probability,
    morris_sensitivity,
    probabilistic_ai_capacity_threshold,
    sobol_sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"


def _set_path(root: dict, path: str, value) -> None:
    target = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[parts[-1]] = value


def make_evaluator(project: dict, timeseries: pd.DataFrame, metric: str):
    def evaluator(parameters: dict) -> float:
        trial = copy.deepcopy(project)
        for path, value in parameters.items():
            _set_path(trial, str(path), value)
        result = FullAIUWMModel(trial, timeseries).run()
        return float(summarize_ai_water_kpis(result, trial)[metric])
    return evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--proposed-mw", type=float, default=1000.0)
    args = parser.parse_args()

    for domain in args.domains.split(","):
        folder = OUT / domain
        folder.mkdir(parents=True, exist_ok=True)
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        timeseries = pd.read_csv(R2 / domain / "timeseries.csv")

        spec = json.loads((R2 / "specs" / "sensitivity_morris.json").read_text(encoding="utf-8"))
        bounds = {name: tuple(map(float, value)) for name, value in spec["parameters"].items()}
        trial = copy.deepcopy(project)
        trial["components"]["AI_DC1"]["installed_it_capacity_mw"] = 1000.0

        print(f"[{domain}] morris ...", flush=True)
        morris = morris_sensitivity(
            make_evaluator(trial, timeseries, spec["metric"]), bounds,
            trajectories=int(spec["trajectories"]), seed=int(spec["seed"]),
        )
        morris.to_csv(folder / "morris.csv", index=False)

        print(f"[{domain}] sobol ...", flush=True)
        sobol_spec = json.loads((R2 / "specs" / "sensitivity_sobol.json").read_text(encoding="utf-8"))
        sobol = sobol_sensitivity(
            make_evaluator(trial, timeseries, sobol_spec["metric"]), bounds,
            samples=int(sobol_spec.get("samples", 32)), seed=int(sobol_spec["seed"]),
        )
        sobol.to_csv(folder / "sobol.csv", index=False)

        print(f"[{domain}] probabilistic ...", flush=True)
        pspec = json.loads(
            (R2 / "specs" / f"probabilistic_threshold_{domain}.json").read_text(encoding="utf-8")
        )
        ranges = {name: tuple(map(float, v)) for name, v in pspec["parameter_ranges"].items()}
        samples = probabilistic_ai_capacity_threshold(
            project, timeseries, ranges, [float(c) for c in pspec["capacities_mw"]],
            {"constraints": pspec["constraints"]},
            samples=int(pspec["samples"]), seed=int(pspec["seed"]),
        )
        samples.to_csv(folder / "probabilistic_thresholds.csv", index=False)
        series = pd.to_numeric(samples["maximum_safe_ai_capacity_mw"], errors="coerce").dropna()
        summary = {
            "samples": int(len(series)),
            "p05": float(series.quantile(0.05)) if len(series) else None,
            "p50": float(series.quantile(0.50)) if len(series) else None,
            "p95": float(series.quantile(0.95)) if len(series) else None,
            "proposed_mw": float(args.proposed_mw),
            "exceedance_probability": float(capacity_exceedance_probability(samples, args.proposed_mw)),
        }
        (folder / "probabilistic_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n=== {domain} ===")
        print(morris.to_string(index=False))
        print(sobol.to_string(index=False))
        print(json.dumps(summary, ensure_ascii=False))
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
