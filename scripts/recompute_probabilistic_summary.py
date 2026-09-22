"""用已有的 probabilistic_large.csv 重算汇总（不重跑模拟）。

用于修正 bootstrap 置信区间的计算错误后刷新 summary，
以及向 summary 补充分布形态与限制约束构成等诊断字段。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "validation_artifacts" / "r2"

R2 = ROOT / "examples" / "cawcc_r2"


def bootstrap_scalar_ci(values: np.ndarray, stat, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        draws[i] = float(stat(values[idx]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    for domain in ("ha", "co"):
        csv_path = OUT / domain / "probabilistic_large.csv"
        old_summary_path = OUT / domain / "probabilistic_large_summary.json"
        spec_path = R2 / "specs" / f"probabilistic_threshold_{domain}.json"
        if not csv_path.exists() or not old_summary_path.exists():
            continue
        old = json.loads(old_summary_path.read_text(encoding="utf-8"))
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        seed = int(spec["seed"])
        bootstrap = int(old.get("bootstrap_iterations", 2000))
        proposed_mw = float(old.get("proposed_mw", 1000.0))

        frame = pd.read_csv(csv_path)
        series = pd.to_numeric(frame["maximum_safe_ai_capacity_mw"], errors="coerce").dropna().to_numpy()

        summary = dict(old)
        for label, quantile in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95)):
            summary[label] = float(np.quantile(series, quantile))
            low, high = bootstrap_scalar_ci(
                series, lambda v, q=quantile: np.quantile(v, q), bootstrap, seed + 1
            )
            summary[f"{label}_ci95"] = [low, high]

        exceed = float((series < proposed_mw).mean())
        exceed_lo, exceed_hi = bootstrap_scalar_ci(
            series, lambda v: float((v < proposed_mw).mean()), bootstrap, seed + 2
        )
        summary["exceedance_probability"] = exceed
        summary["exceedance_probability_ci95"] = [exceed_lo, exceed_hi]
        summary["mean"] = float(series.mean())
        summary["std"] = float(series.std(ddof=1))
        summary["distinct_thresholds"] = sorted({float(v) for v in series})
        share = frame["limiting_constraint"].value_counts(normalize=True).round(4).to_dict()
        summary["limiting_constraint_share"] = share
        dist = frame["maximum_safe_ai_capacity_mw"].value_counts(normalize=True).sort_index().round(4)
        summary["threshold_distribution"] = {f"{float(k):.0f}": float(v) for k, v in dist.items()}

        old_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{domain}] n={len(series)} 超限概率={exceed:.4f} "
              f"CI95=[{exceed_lo:.4f}, {exceed_hi:.4f}] 约束构成={share}")


if __name__ == "__main__":
    main()
