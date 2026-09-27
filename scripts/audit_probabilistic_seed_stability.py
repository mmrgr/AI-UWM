"""Cross-seed stability check for the probabilistic capacity batch.

Motivation: the bootstrap CI in ``probabilistic_corrected_summary.json`` only
measures sampling noise *inside* one 1024-draw batch.  A narrow CI therefore
does not prove that the underlying distribution has converged.  This script
runs the same specification under an independent random seed and compares the
quantiles, the threshold-exceedance probability and the limiting-constraint
attribution.

The production outputs are never touched: everything runs in a temporary copy
under ``.seedcheck/iso``.

Usage::

    python scripts/audit_probabilistic_seed_stability.py --samples 128
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "validation_artifacts" / "r2"
ISO = ROOT / ".seedcheck" / "iso"
ARCHIVED = {
    "ha": R2 / "probabilistic_seedcheck_77700001_ha_summary.json",
    "co": R2 / "probabilistic_seedcheck_77700001_co_summary.json",
}
# Keys whose stability matters for the paper's conclusions.
COMPARED = [
    "p05",
    "p50",
    "p95",
    "exceedance_probability",
    "limiting_constraint_share",
]


def prepare_isolated_copy(seed: int) -> None:
    if ISO.exists():
        shutil.rmtree(ISO)
    ISO.mkdir(parents=True)
    for name in ("src", "scripts", "examples"):
        shutil.copytree(ROOT / name, ISO / name)
    # Rewrite the seed in the isolated specs so the reference run is untouched.
    for domain in ("ha", "co"):
        spec_path = ISO / "examples" / "cawcc_r2" / "specs" / f"probabilistic_threshold_{domain}.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["seed"] = seed
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
    (ISO / "validation_artifacts" / "r2").mkdir(parents=True)


def run_reference(samples: int, workers: int, bootstrap: int) -> None:
    env = {"PYTHONPATH": "src;scripts"}
    cmd = [
        sys.executable,
        "scripts/run_r2_probabilistic.py",
        "--domains",
        "ha,co",
        "--samples",
        str(samples),
        "--workers",
        str(workers),
        "--bootstrap",
        str(bootstrap),
    ]
    subprocess.run(cmd, cwd=ISO, env={**env}, check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(domain: str, reference: dict, candidate: dict) -> list[dict]:
    """Compare two batches.

    Tolerance policy (why exact equality is the wrong test):

    * ``p50`` / ``p95`` of the physical CAWCC must be identical — they are the
      centrel and upper tail the paper quotes.
    * ``p05`` sits on a steep part of the threshold CDF and may move by one
      50 MW grid step.
    * ``exceedance_probability`` is bounded by 1/n; two independent draws can
      differ by a few Bernoulli counts without meaning a distribution shift.
    * ``limiting_constraint_share`` is a share over a finite sample, so it is
      compared on the *dominant* constraint only, not on the exact percentage.
    """
    rows: list[dict] = []

    def add(stat: str, base, new, ok: bool, note: str = "") -> None:
        rows.append(
            {
                "域": domain,
                "统计量": stat,
                "参考批次": json.dumps(base, ensure_ascii=False, sort_keys=True),
                "独立种子": json.dumps(new, ensure_ascii=False, sort_keys=True),
                "一致": ok,
                "判据": note,
            }
        )

    # Centre and upper tail: must be exactly equal.
    for key in ("p50", "p95"):
        add(key, reference.get(key), candidate.get(key), reference.get(key) == candidate.get(key), "必须相等")

    # Lower tail: allowed to move by at most one grid step.
    base05, new05 = reference.get("p05"), candidate.get("p05")
    ok05 = base05 is not None and new05 is not None and abs(base05 - new05) <= 50.0
    add("p05", base05, new05, ok05, "允许 ≤1 个 50 MW 网格步")

    # Exceedance probability: same ballpark, judged by the pooled 95% interval.
    base_p, new_p = reference.get("exceedance_probability"), candidate.get("exceedance_probability")
    base_ci = reference.get("exceedance_probability_ci95") or [0.0, 1.0]
    ok_p = base_p is not None and new_p is not None and base_ci[0] - 0.02 <= new_p <= base_ci[1] + 0.02
    add("exceedance_probability", base_p, new_p, ok_p, "须落在参考批次 95% 区间内（容差 0.02）")

    # Attribution: the dominant limiting constraint must not change.
    base_l = reference.get("limiting_constraint_share") or {}
    new_l = candidate.get("limiting_constraint_share") or {}
    base_top = max(base_l, key=base_l.get) if base_l else None
    new_top = max(new_l, key=new_l.get) if new_l else None
    add(
        "limiting_constraint_share",
        base_l,
        new_l,
        base_top is not None and base_top == new_top,
        "主导约束不得改变（比例允许抽样波动）",
    )

    base_pol = reference.get("policy_threshold", {})
    new_pol = candidate.get("policy_threshold", {})
    for key in ("p05", "p50", "p95"):
        add(f"policy_threshold.{key}", base_pol.get(key), new_pol.get(key), base_pol.get(key) == new_pol.get(key), "必须相等")
    rc_base, rc_new = base_pol.get("right_censored_at_grid_max_fraction"), new_pol.get("right_censored_at_grid_max_fraction")
    add(
        "policy_threshold.right_censored_at_grid_max_fraction",
        rc_base,
        rc_new,
        rc_base is not None and rc_new is not None and abs(rc_base - rc_new) <= 0.05,
        "删失率为有限样本比例，容差 0.05",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=77700001)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--reuse", action="store_true", help="复用已归档的独立种子结果，不重跑")
    args = parser.parse_args()

    if not args.reuse:
        print(f"在隔离副本中用 seed={args.seed} 重跑 {args.samples} 样本 ...")
        prepare_isolated_copy(args.seed)
        run_reference(args.samples, args.workers, args.bootstrap)

    rows: list[dict] = []
    for domain in ("ha", "co"):
        reference = load(R2 / domain / "probabilistic_corrected_summary.json")
        candidate_path = ISO / "validation_artifacts" / "r2" / domain / "probabilistic_corrected_summary.json"
        if candidate_path.exists():
            candidate = load(candidate_path)
        elif ARCHIVED[domain].exists():
            candidate = load(ARCHIVED[domain])
        else:
            print(f"缺少 {domain} 的独立种子结果，先不加 --reuse 运行一次")
            return 1
        rows.extend(compare(domain, reference, candidate))

    out = R2 / "probabilistic_seed_stability.csv"
    import csv

    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["域", "统计量", "参考批次", "独立种子", "一致", "判据"])
        writer.writeheader()
        writer.writerows(rows)

    unstable = [r for r in rows if not r["一致"]]
    print(f"对照 {len(rows)} 项 -> {out}")
    for r in rows:
        flag = "OK  " if r["一致"] else "FAIL"
        print(f"  {flag} {r['域']} {r['统计量']}: {r['参考批次']} vs {r['独立种子']}  [{r['判据']}]")

    if unstable:
        print(f"\nFAIL: {len(unstable)} 项超出容差")
        return 1
    print("\nOK: 中心位置、上尾、限制约束归因与政策阈值跨种子稳定")
    return 0


if __name__ == "__main__":
    sys.exit(main())
