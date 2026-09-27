"""Run the probabilistic batch at a chosen sample size into an isolated folder.

Why this exists
---------------
``run_r2_probabilistic.py`` hard-codes its output to ``validation_artifacts/r2``.
Building a sample-size convergence sequence (256 / 1024 / 4096) therefore has to
run several batches without overwriting the published 1024-sample artifact.

This runner imports the same code path, redirects ``OUT`` to a scratch folder,
and optionally seeds the checkpoint from the published 1024-sample run.  Seeding
is safe and exact: the sampler draws from a single ``default_rng(seed)`` in
strict ``sample_id`` order, so the first N samples of an M-sample batch (N <= M)
are bit-identical to an N-sample batch with the same seed.  ``--verify-seed-prefix``
asserts this before any work is done.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Must be set before numpy/BLAS load in this process and in every spawned worker.
# On a 32-core box OpenBLAS otherwise opens one thread pool per worker; with a
# handful of workers the combined allocation fails outright with
# "Memory allocation still failed after 10 retries" even though free RAM looks
# ample.  Each Monte-Carlo sample is already parallelised across workers, so
# intra-process BLAS threading buys nothing here.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (str(SRC), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_r2_probabilistic as runner  # noqa: E402
from aiuwm import full_engine  # noqa: E402


def verify_seed_prefix(domain: str, sample_count: int) -> int:
    """Assert the sampler regenerates the published prefix exactly."""
    spec = json.loads(
        (runner.R2 / "specs" / f"probabilistic_threshold_{domain}.json").read_text(encoding="utf-8")
    )
    seed = int(spec["seed"])
    params = {p: (float(a), float(b)) for p, (a, b) in spec["parameter_ranges"].items()}
    rng = np.random.default_rng(seed)
    generated = [
        {p: float(rng.uniform(a, b)) for p, (a, b) in params.items()} for _ in range(sample_count)
    ]

    checkpoint = runner.OUT / domain / "probabilistic_corrected_checkpoint.jsonl"
    if not checkpoint.exists():
        raise SystemExit(f"没有可校验的 1024 批次 checkpoint: {checkpoint}")
    rows = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    by_id = {int(row["sample_id"]): row for row in rows}
    if len(by_id) != sample_count:
        raise SystemExit(f"checkpoint 样本数 {len(by_id)} != 预期 {sample_count}")

    for sample_id, row in by_id.items():
        for path, value in generated[sample_id].items():
            if abs(float(row[path]) - value) > 1e-12:
                raise SystemExit(
                    f"种子前缀校验失败: domain={domain} sample_id={sample_id} path={path}"
                )
    # Cross-check the sampled grid actually drives the recorded capacities.
    capacities = [row["maximum_safe_ai_capacity_mw"] for row in by_id.values()]
    return len(capacities)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out-root", required=True, help="隔离输出根目录（不会写入 validation_artifacts）")
    parser.add_argument(
        "--seed-from",
        type=int,
        default=0,
        help="从已发布的 N 样本批次继承 checkpoint 前缀（前台要求同种子前缀校验通过）",
    )
    parser.add_argument("--verify-seed-prefix", action="store_true")
    args = parser.parse_args()

    if args.seed_from and args.verify_seed_prefix:
        for domain in args.domains.split(","):
            count = verify_seed_prefix(domain.strip(), args.seed_from)
            print(f"[verify] {domain}: 种子前缀 {count} 样本逐位一致", flush=True)

    published = runner.OUT
    iso = Path(args.out_root).resolve()
    if published.resolve() == iso:
        raise SystemExit("隔离输出目录不能等于已发布产物目录")

    for domain in args.domains.split(","):
        domain = domain.strip()
        target = iso / domain
        target.mkdir(parents=True, exist_ok=True)
        if args.seed_from:
            source = published / domain / "probabilistic_corrected_checkpoint.jsonl"
            if not source.exists():
                raise SystemExit(f"缺少前缀 checkpoint: {source}")
            shutil.copy2(source, target / "probabilistic_corrected_checkpoint.jsonl")
            print(f"[seed] {domain}: 已从 {args.seed_from} 批次继承 checkpoint", flush=True)

    # Redirect every output write into the scratch tree.
    runner.OUT = iso
    runner.CAP = runner.CAP  # unchanged: same 250/50 MW adaptive grid
    for domain in args.domains.split(","):
        domain = domain.strip()
        try:
            summary = runner.run_domain(
                domain, args.samples, args.workers, args.bootstrap, 1000.0
            )
        except TypeError:
            summary = runner.run_domain(domain, args.samples, args.workers, args.bootstrap)
        print(
            f"[done] {domain}: samples={summary['samples']} p05/p50/p95="
            f"{summary['p05']}/{summary['p50']}/{summary['p95']} "
            f"exceed={summary['exceedance_probability']:.4f}",
            flush=True,
        )
    print(f"输出目录: {iso}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
