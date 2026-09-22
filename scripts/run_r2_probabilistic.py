"""R2 大样本概率承载阈值（任务 B）。

原 probabilistic_threshold spec 只取 40 个样本，用 40 个样本去估 P05/P95
在统计上不可靠（P05 落在第 2 个样本上）。本脚本把样本量提到 1024，并用
bootstrap 给出分位数的 95% 置信区间。

关键实现：
  * 主进程先用同一个 seed 预采样全部参数向量，再分发 worker。
    -> 与串行调用 probabilistic_ai_capacity_threshold 的结果逐位一致，可复现。
  * ProcessPoolExecutor 并行（每样本要对全部容量点做一次扫描，单核不可行）。

产出（写回原 domain 产物目录）:
  probabilistic_large.csv            1024 个样本的原始记录
  probabilistic_large_summary.json   分位数 + bootstrap 95% CI + 超限概率

用法:
    python scripts/run_r2_probabilistic.py --samples 1024 --workers 24
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"

CAP: list[float] = []
CONSTRAINTS: dict = {}
_PARAMS: dict[str, tuple[float, float]] = {}


def _import_engine():
    """Windows spawn 出来的子进程不带 PYTHONPATH，必须显式补上 src。"""
    candidate = str(Path(sys.argv[0]).resolve().parents[1] / "src")
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    from aiuwm.ai_capacity import find_ai_carrying_capacity, scan_ai_capacity
    from aiuwm.sensitivity import _set_parameter

    return find_ai_carrying_capacity, scan_ai_capacity, _set_parameter


def _set_path(root: dict, path: str, value) -> None:
    _, _, set_parameter = _import_engine()
    set_parameter(root, path, value)


_WORKER_STATE: dict = {}


def _init_worker(project: dict, timeseries: pd.DataFrame, params: dict, cap: list, constraints: dict) -> None:
    """Windows 用 spawn，子进程不继承主进程内存，所有常量必须显式传入。"""
    _WORKER_STATE["project"] = project
    _WORKER_STATE["ts"] = timeseries
    _WORKER_STATE["params"] = params
    _WORKER_STATE["cap"] = cap
    _WORKER_STATE["constraints"] = constraints


def _run_sample(job: tuple[int, dict]) -> dict:
    """跑单个样本：按给定参数向量设定 -> 容量扫描 -> 判定 CAWCC。"""
    find_ai_carrying_capacity, scan_ai_capacity, _ = _import_engine()
    sample_id, sampled = job
    trial = copy.deepcopy(_WORKER_STATE["project"])
    for path, value in sampled.items():
        _set_path(trial, path, value)
    scan = scan_ai_capacity(trial, _WORKER_STATE["ts"], capacities_mw=_WORKER_STATE["cap"])
    threshold = find_ai_carrying_capacity(scan, _WORKER_STATE["constraints"])
    row = {"sample_id": sample_id}
    row.update(sampled)
    row.update({k: v for k, v in threshold.items() if k != "capacity_scan"})
    return row


def _bootstrap_scalar_ci(values: np.ndarray, stat, iterations: int, seed: int) -> tuple[float, float]:
    """对任意标量统计量（均值、比例等）做 bootstrap 百分位 CI。

    注意：不能用 _bootstrap_quantile_ci(..., 0.5) 来代替——对 0/1 指示序列取
    中位数会得到恒为 0 的退化区间。
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        draws[i] = float(stat(values[idx]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def run_domain(domain: str, samples: int, workers: int, bootstrap: int, proposed_mw: float) -> dict:
    project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
    spec = json.loads(
        (R2 / "specs" / f"probabilistic_threshold_{domain}.json").read_text(encoding="utf-8")
    )

    global CAP, CONSTRAINTS
    CAP = [float(c) for c in spec["capacities_mw"]]
    CONSTRAINTS = spec["constraints"]
    params = {path: (float(low), float(high)) for path, (low, high) in spec["parameter_ranges"].items()}
    seed = int(spec["seed"])
    # 主进程预采样：与串行路径用同一 seed + 同一 rng 调用顺序 -> 逐位可复现
    rng = np.random.default_rng(seed)
    jobs: list[tuple[int, dict]] = []
    for sample_id in range(samples):
        sampled = {path: float(rng.uniform(low, high)) for path, (low, high) in params.items()}
        jobs.append((sample_id, sampled))

    print(f"[{domain}] {samples} 样本 x {len(CAP)} 容量点, workers={workers}", flush=True)
    started = time.time()
    rows: list[dict] = []
    chunk = max(1, samples // (workers * 4))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(project, timeseries, params, CAP, CONSTRAINTS),
    ) as pool:
        for index, row in enumerate(pool.map(_run_sample, jobs, chunksize=chunk), start=1):
            rows.append(row)
            if index % 100 == 0 or index == samples:
                print(f"  {index:5d}/{samples}  {time.time() - started:6.1f}s", flush=True)
    elapsed = time.time() - started

    frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    folder = OUT / domain
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "probabilistic_large.csv", index=False)

    series = pd.to_numeric(frame["maximum_safe_ai_capacity_mw"], errors="coerce").dropna().to_numpy()
    summary: dict = {
        "samples": int(len(series)),
        "workers": int(workers),
        "elapsed_seconds": round(elapsed, 1),
        "seed": seed,
        "proposed_mw": float(proposed_mw),
        "bootstrap_iterations": int(bootstrap),
    }
    for label, quantile in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95)):
        point = float(np.quantile(series, quantile))
        low, high = _bootstrap_scalar_ci(
            series, lambda v, q=quantile: np.quantile(v, q), bootstrap, seed + 1
        )
        summary[label] = point
        summary[f"{label}_ci95"] = [low, high]

    exceed = float((series < proposed_mw).mean())
    exceed_lo, exceed_hi = _bootstrap_scalar_ci(
        series, lambda v: float((v < proposed_mw).mean()), bootstrap, seed + 2
    )
    summary["exceedance_probability"] = exceed
    summary["exceedance_probability_ci95"] = [float(exceed_lo), float(exceed_hi)]
    summary["mean"] = float(series.mean())
    summary["std"] = float(series.std(ddof=1))
    summary["distinct_thresholds"] = sorted({float(v) for v in series})
    constraint_counts = frame["limiting_constraint"].value_counts().to_dict()
    summary["limiting_constraint_share"] = {
        k: round(v / len(frame), 4) for k, v in constraint_counts.items()
    }

    (folder / "probabilistic_large_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{domain}] P05={summary['p05']:.0f} P50={summary['p50']:.0f} P95={summary['p95']:.0f} "
        f"| 超限概率={exceed:.3f} | 用时 {elapsed:.0f}s",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--proposed-mw", type=float, default=1000.0)
    args = parser.parse_args()

    previous = {}
    for domain in args.domains.split(","):
        old = OUT / domain.strip() / "probabilistic_summary.json"
        if old.exists():
            previous[domain.strip()] = json.loads(old.read_text(encoding="utf-8"))

    for domain in args.domains.split(","):
        run_domain(domain.strip(), args.samples, args.workers, args.bootstrap, args.proposed_mw)

    # 与 40 样本版本的对照（方法学观察：小样本分位数估计的偏差）
    comparison = []
    for domain in previous:
        new_path = OUT / domain / "probabilistic_large_summary.json"
        if not new_path.exists():
            continue
        new = json.loads(new_path.read_text(encoding="utf-8"))
        comparison.append({
            "域": domain,
            "版本": "40 样本",
            "P05": previous[domain].get("p05"),
            "P50": previous[domain].get("p50"),
            "P95": previous[domain].get("p95"),
            "超限概率": previous[domain].get("exceedance_probability"),
        })
        comparison.append({
            "域": domain,
            "版本": f"{new['samples']} 样本",
            "P05": new.get("p05"),
            "P50": new.get("p50"),
            "P95": new.get("p95"),
            "超限概率": new.get("exceedance_probability"),
        })
    if comparison:
        frame = pd.DataFrame(comparison)
        frame.to_csv(OUT / "probabilistic_sample_size_comparison.csv", index=False, encoding="utf-8-sig")
        print(f"\n样本量对照 -> {OUT / 'probabilistic_sample_size_comparison.csv'}")
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
