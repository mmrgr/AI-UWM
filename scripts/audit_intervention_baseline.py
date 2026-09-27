"""Reconcile the intervention baseline against the main carrying-capacity results.

The cost/Pareto pipeline (``run_r2_cost_pareto.py``) deliberately normalises every
intervention to one baseline per domain so that gains are comparable across
interventions (HA=1100, CO=1400 MW).  The main threshold artifact reports six
separate estimands per domain.  Readers who see "+200 MW" in the intervention
table and "+150 MW" in the narrative are looking at two different baselines, not
at two contradictory measurements.

This script does not re-run the model.  It reads the existing artifacts and
emits a single table that shows, for each intervention, the gain measured from
the intervention-table baseline and from the headline daily physical CAWCC, so
the two numbers can be reconciled explicitly rather than left implicit.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "validation_artifacts" / "r2"
OUT = R2 / "intervention_baseline_alignment.csv"


def load(domain: str) -> tuple[dict, list[dict], dict]:
    threshold = json.loads(
        (R2 / domain / "capacity_threshold_corrected.json").read_text(encoding="utf-8")
    )
    with (R2 / domain / "intervention_marginal.csv").open(encoding="utf-8-sig") as fh:
        marginal = list(csv.DictReader(fh))
    cost = {}
    with (R2 / "intervention_cost_benefit.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["域"] == domain:
                cost[row["干预"]] = row
    return threshold, marginal, cost


def main() -> int:
    rows: list[dict] = []
    for domain in ("ha", "co"):
        threshold, marginal, cost = load(domain)
        by_name = {r["intervention"]: r for r in marginal}
        baseline = by_name["baseline"]
        table_baseline = float(baseline["maximum_safe_ai_capacity_mw"])
        daily_safe = float(threshold["physical_daily"]["maximum_safe_ai_capacity_mw"])
        hourly_safe = float(threshold["physical_hourly"]["maximum_safe_ai_capacity_mw"])

        for name, record in by_name.items():
            post = float(record["maximum_safe_ai_capacity_mw"])
            rows.append(
                {
                    "域": domain,
                    "干预": name,
                    "干预后CAWCC_MW": post,
                    "干预表基线_MW": table_baseline,
                    "干预表口径增益_MW": round(post - table_baseline, 3),
                    "物理日尺度基线_MW": daily_safe,
                    "物理日尺度口径增益_MW": round(post - daily_safe, 3),
                    "小时代理基线_MW": hourly_safe,
                    "小时代理口径增益_MW": round(post - hourly_safe, 3),
                    "限制约束": record["limiting_constraint"],
                    "单位成本_万元每MW": cost.get(name, {}).get("单位成本_万元每MW", ""),
                }
            )

    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"written -> {OUT}")
    for domain in ("ha", "co"):
        print(f"\n=== {domain.upper()} ===")
        for r in rows:
            if r["域"] != domain:
                continue
            print(
                f"  {r['干预']:<28} 后={r['干预后CAWCC_MW']:>6.0f}  "
                f"表口径={r['干预表口径增益_MW']:>+7.1f}  "
                f"日尺度口径={r['物理日尺度口径增益_MW']:>+7.1f}  "
                f"小时口径={r['小时代理口径增益_MW']:>+7.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
