"""Audit adaptive probabilistic capacity grids against the full 50 MW grid.

The production probability run uses 250 MW anchors and restores 50 MW points
around detected transitions.  This script checks that shortcut on eight fixed
parameter draws spanning the seeded 1024-draw prior for each domain.
"""
from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2_probabilistic as rp  # noqa: E402

R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2" / "probabilistic_grid_audit_8.json"
SAMPLE_IDS = [0, 1, 127, 255, 383, 511, 767, 1023]


def _sample_vectors(spec: dict, count: int = 1024) -> list[dict[str, float]]:
    params = {
        path: (float(low), float(high))
        for path, (low, high) in spec["parameter_ranges"].items()
    }
    rng = np.random.default_rng(int(spec["seed"]))
    rows = []
    for _ in range(count):
        rows.append({path: float(rng.uniform(low, high)) for path, (low, high) in params.items()})
    return rows


def _run_one(job: tuple[int, dict[str, float]]) -> dict:
    sample_id, sampled = job
    _, scan_ai_capacity, _ = rp._import_engine()
    trial = copy.deepcopy(rp._WORKER_STATE["project"])
    for path, value in sampled.items():
        rp._set_path(trial, path, value)
    capacities = rp._WORKER_STATE["cap"]
    constraints = rp._WORKER_STATE["constraints"]
    policy = rp._WORKER_STATE["policy_constraints"]
    full = scan_ai_capacity(trial, rp._WORKER_STATE["ts"], capacities_mw=capacities)
    full_physical = rp._import_engine()[0](full, constraints)
    full_policy = rp._import_engine()[0](
        full.loc[full["ai_capacity_mw"] > 0], policy
    )
    adaptive = rp._adaptive_scan(trial, capacities, constraints, policy)
    adaptive_physical = rp._import_engine()[0](adaptive, constraints)
    adaptive_policy = rp._import_engine()[0](
        adaptive.loc[adaptive["ai_capacity_mw"] > 0], policy
    )
    return {
        "sample_id": sample_id,
        "physical_full_mw": full_physical["maximum_safe_ai_capacity_mw"],
        "physical_adaptive_mw": adaptive_physical["maximum_safe_ai_capacity_mw"],
        "policy_full_mw": full_policy["maximum_safe_ai_capacity_mw"],
        "policy_adaptive_mw": adaptive_policy["maximum_safe_ai_capacity_mw"],
        "full_grid_points": int(len(full)),
        "adaptive_grid_points": int(len(adaptive)),
    }


def run_domain(domain: str, workers: int) -> dict:
    project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
    spec = json.loads(
        (R2 / "specs" / f"probabilistic_threshold_{domain}.json").read_text(encoding="utf-8")
    )
    capacities = [float(value) for value in spec["capacities_mw"]]
    constraints = spec["constraints"]
    policy = spec.get("policy_constraints", {})
    vectors = _sample_vectors(spec)
    jobs = [(sample_id, vectors[sample_id]) for sample_id in SAMPLE_IDS]
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=rp._init_worker,
        initargs=(project, timeseries, spec["parameter_ranges"], capacities, constraints, policy),
    ) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["sample_id"])
    return {"domain": domain, "checked_samples": rows}


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    reports = [run_domain(domain.strip(), args.workers) for domain in args.domains.split(",")]
    checks = [row for report in reports for row in report["checked_samples"]]
    all_match = all(
        row["physical_full_mw"] == row["physical_adaptive_mw"]
        and row["policy_full_mw"] == row["policy_adaptive_mw"]
        for row in checks
    )
    payload = {
        "method": "full 50 MW grid versus adaptive 250 MW anchors with transition refinement",
        "sample_ids": SAMPLE_IDS,
        "domains": reports,
        "all_match": bool(all_match),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
