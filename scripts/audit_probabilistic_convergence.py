"""Formal sample-size convergence diagnostics for the probabilistic CAWCC.

The published 1024-sample batch reports bootstrap intervals that are narrow
(HA P50 width 50 MW), but narrow bootstrap intervals only describe within-batch
sampling noise.  They do not establish that the estimate has converged as the
number of Monte-Carlo samples grows.  The R5 plan (T0.3) requires an explicit
convergence judgement on a same-seed sequence: 256 / 1024 / 4096.

This script reads the per-sample capacity series from each batch, then reports:

* quantile trajectories across sample sizes (do P05/P50/P95 stabilise?);
* the standard error of the mean at each size, and its log-log slope.  For an
  independent sampler the slope should be approximately -1/2; a slope that is
  materially shallower indicates the estimator is not yet in the asymptotic
  regime;
* exceedance-probability trajectories with binomial standard errors;
* a pass/fail against the pre-declared gate: every quantile must move by no
  more than one 50 MW grid step from the largest batch, and the mean's standard
  error must shrink monotonically with sample size.

Inputs are the isolated batch directories produced by
``run_r5_convergence_batch.py``; the published 1024 batch is read from
``validation_artifacts/r2``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "validation_artifacts" / "r2"

# Pre-declared acceptance thresholds (stated before looking at the numbers).
QUANTILE_TOLERANCE_MW = 50.0
MIN_ABS_SLOPE = 0.30  # |slope| should be at least this close to the ideal 0.5
GRID_STEP_MW = 50.0


def load_series(path: Path) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(path)
    return pd.to_numeric(frame["maximum_safe_ai_capacity_mw"], errors="coerce").dropna().to_numpy()


def load_checkpoint_series(path: Path) -> np.ndarray:
    """Read a partially completed batch straight from its checkpoint.

    A run that is stopped early never writes ``probabilistic_corrected.csv``,
    but the checkpoint still holds every finished sample keyed by ``sample_id``.
    Because the sampler emits samples in strict ``sample_id`` order, the first
    N records of a longer run are exactly the first N samples of the seed
    sequence -- so a partial batch is a legitimate, if shorter, prefix point.
    """
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_id = {int(r["sample_id"]): r for r in records}
    ordered = [by_id[k]["maximum_safe_ai_capacity_mw"] for k in sorted(by_id)]
    return np.asarray(ordered, dtype=float)


def collect(domain: str, scratch: Path) -> dict[int, tuple[Path, bool]]:
    """Map sample size -> (series file, is_partial), from every available batch.

    Two batches can land on the same sample count -- e.g. a partially completed
    4096 run whose inherited prefix happens to equal the published 1024 batch.
    A completed batch is always preferred over a partial one at the same size,
    otherwise a prefix-only checkpoint would mask the real published result.
    """
    found: dict[int, tuple[Path, bool]] = {}

    def offer(size: int, path: Path, is_partial: bool) -> None:
        existing = found.get(size)
        if existing is None or (existing[1] and not is_partial):
            found[size] = (path, is_partial)

    published = PUBLISHED / domain / "probabilistic_corrected.csv"
    if published.exists():
        offer(1024, published, False)
    if scratch.exists():
        import pandas as pd

        for folder in sorted(scratch.glob("iso*")):
            candidate = folder / domain / "probabilistic_corrected.csv"
            if candidate.exists():
                offer(len(pd.read_csv(candidate)), candidate, False)
                continue
            checkpoint = folder / domain / "probabilistic_corrected_checkpoint.jsonl"
            if checkpoint.exists():
                series = load_checkpoint_series(checkpoint)
                if len(series):
                    offer(len(series), checkpoint, True)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--scratch", required=True, help="隔离批次根目录（含 iso256/iso4096）")
    parser.add_argument("--proposed-mw", type=float, default=1000.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scratch = Path(args.scratch).resolve()
    report: dict = {
        "gate": {
            "quantile_tolerance_mw": QUANTILE_TOLERANCE_MW,
            "min_abs_log_log_slope": MIN_ABS_SLOPE,
            "note": "阈值在看数之前预先声明；未达标则只能表述为条件于最大批次的估计",
        },
        "domains": {},
    }

    for domain in args.domains.split(","):
        domain = domain.strip()
        batches = collect(domain, scratch)
        if len(batches) < 3:
            report["domains"][domain] = {
                "status": "INSUFFICIENT_BATCHES",
                "available_sample_sizes": sorted(batches),
                "note": "至少需要 3 个样本量点才能做收敛判定",
                "statement": "批次不足，暂不能作收敛判定；只能表述为'条件于现有批次的估计'",
            }
            continue

        sizes = sorted(batches)
        points: list[dict] = []
        partial_sizes: list[int] = []
        for size in sizes:
            path, is_partial = batches[size]
            series = load_series(path) if not is_partial else load_checkpoint_series(path)
            if is_partial:
                partial_sizes.append(size)
            n = len(series)
            sd = float(series.std(ddof=1)) if n > 1 else 0.0
            points.append(
                {
                    "samples": n,
                    "partial_batch": bool(is_partial),
                    "p05": float(np.quantile(series, 0.05)),
                    "p50": float(np.quantile(series, 0.50)),
                    "p95": float(np.quantile(series, 0.95)),
                    "mean": float(series.mean()),
                    "std": sd,
                    "mean_standard_error": sd / math.sqrt(n) if n else None,
                    "exceedance_probability": float((series < args.proposed_mw).mean()),
                    "exceedance_standard_error": float(
                        math.sqrt(
                            max((series < args.proposed_mw).mean()
                                * (1 - (series < args.proposed_mw).mean()), 0.0) / n
                        )
                    )
                    if n
                    else None,
                }
            )

        # Log-log slope of the mean's standard error versus sample size.
        usable = [p for p in points if p["mean_standard_error"] and p["mean_standard_error"] > 0]
        slope = None
        if len(usable) >= 2:
            x = np.log([p["samples"] for p in usable])
            y = np.log([p["mean_standard_error"] for p in usable])
            slope = float(np.polyfit(x, y, 1)[0])

        largest = points[-1]
        checks = {
            "quantiles_within_tolerance": all(
                abs(p[q] - largest[q]) <= QUANTILE_TOLERANCE_MW
                for p in points
                for q in ("p05", "p50", "p95")
            ),
            "mean_se_monotone_decreasing": all(
                usable[i]["mean_standard_error"] >= usable[i + 1]["mean_standard_error"] - 1e-12
                for i in range(len(usable) - 1)
            )
            if len(usable) >= 2
            else False,
            "abs_slope_ge_threshold": bool(slope is not None and abs(slope) >= MIN_ABS_SLOPE),
        }
        passed = all(checks.values())

        if passed and not partial_sizes:
            statement = "可按'已通过同种子跨样本量收敛判定'表述"
        elif passed and partial_sizes:
            statement = (
                "预声明门槛已通过，但最大批次为部分完成批次（"
                + ", ".join(str(s) for s in partial_sizes)
                + "），只能表述为'在已完成样本量范围内稳定'"
            )
        else:
            statement = "只能表述为'条件于最大批次的估计'，不得声称已收敛"

        report["domains"][domain] = {
            "status": "CONVERGED" if (passed and not partial_sizes)
            else ("CONVERGED_PARTIAL" if passed else "NOT_CONVERGED"),
            "available_sample_sizes": sizes,
            "partial_batch_sizes": partial_sizes,
            "points": points,
            "mean_se_log_log_slope": slope,
            "checks": checks,
            "statement": statement,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written -> {out}")
    for domain, block in report["domains"].items():
        print(f"\n=== {domain} : {block['status']} ===")
        for p in block.get("points", []):
            print(
                f"  n={p['samples']:>5}  P05/P50/P95={p['p05']:.0f}/{p['p50']:.0f}/{p['p95']:.0f}"
                f"  SE={p['mean_standard_error']:.2f}  exc={p['exceedance_probability']:.4f}"
            )
        if block.get("mean_se_log_log_slope") is not None:
            print(f"  log-log SE 斜率 = {block['mean_se_log_log_slope']:.3f} (理想 -0.5)")
        print(f"  {block['statement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
