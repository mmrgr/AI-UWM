"""逐点违反诊断：在 D-HA / D-CO 的边界容量附近，逐条打印每个日尺度与年尺度约束的
取值与阈值，并明确指出**第一个真正越界的约束**。

用途：核查 10.2 / 12.3 / 12.5 节反复出现的"限制约束是 reclaimed_water_substitution_ratio"
是否成立，以及它是否属于"阈值贴着配置值"造成的**抢占式伪约束**。

用法:
    PYTHONPATH="src;scripts" python scripts/diag_constraint_binding.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aiuwm.ai_metrics import summarize_ai_water_kpis  # noqa: E402
from aiuwm.full_engine import FullAIUWMModel  # noqa: E402
from run_r2 import R2, annual_constraints, daily_constraints, mean_utilizations  # noqa: E402

OUT = ROOT / "validation_artifacts" / "r2"


def evaluate(project, timeseries, constraints, capacity) -> dict:
    cfg = json.loads(json.dumps(project))
    for c in cfg["components"].values():
        if c.get("kind") == "data_center":
            c["installed_it_capacity_mw"] = float(capacity)
            c.pop("capacity_schedule", None)
    result = FullAIUWMModel(cfg, timeseries).run()
    kpi = summarize_ai_water_kpis(result, cfg)
    util = mean_utilizations(result, cfg)
    return {**kpi, **util}


def violated(name: str, value, definition: dict) -> bool | None:
    if value is None or value != value:
        return None
    op = definition.get("operator")
    thr = float(definition.get("value"))
    if op == ">=":
        return value < thr
    if op == "<=":
        return value > thr
    return None


def main() -> None:
    rows = []
    for domain in ("ha", "co"):
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        timeseries = pd.read_csv(R2 / domain / "timeseries.csv")
        raw = json.loads(
            (R2 / "specs" / f"constraints_{domain}.json").read_text(encoding="utf-8")
        )

        grids = {
            "日尺度": daily_constraints(raw),
            "年尺度": annual_constraints(raw),
        }

        print(f"\n{'=' * 78}\n{domain.upper()}  逐点约束诊断\n{'=' * 78}", flush=True)
        for scale, defs in grids.items():
            print(f"\n--- {scale} ---")
            header = f"{'容量':>6} | " + " | ".join(
                f"{k[:26]:>26}" for k in defs
            )
            print(header)
            print(f"{'阈值':>6} | " + " | ".join(
                f"{defs[k].get('operator')}{defs[k].get('value')}"[-26:] for k in defs
            ))
            for cap in np.arange(900, 1501, 50, dtype=float):
                rec = evaluate(project, timeseries, raw, cap)
                cells = []
                for k in defs:
                    v = rec.get(k)
                    if v is None or v != v:
                        cells.append(f"{'--':>26}")
                        continue
                    bad = violated(k, v, defs[k])
                    mark = "!" if bad else " "
                    cells.append(f"{v:>25.4f}{mark}")
                print(f"{cap:>6.0f} | " + " | ".join(cells))

                # 记录第一个真正越界的约束
                first = [k for k in defs if violated(k, rec.get(k), defs[k])]
                rows.append({
                    "域": domain, "尺度": scale, "容量_MW": cap,
                    "越界约束数": len(first),
                    "越界约束": "; ".join(first),
                    **{k: rec.get(k) for k in defs},
                })

    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "constraint_binding_diagnosis.csv", index=False, encoding="utf-8-sig")

    print("\n\n=== 每个 (域, 尺度) 第一个出现越界的容量 ===")
    summary = (
        frame[frame["越界约束数"] > 0]
        .groupby(["域", "尺度"])
        .first()[["容量_MW", "越界约束数", "越界约束"]]
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"\n-> {OUT / 'constraint_binding_diagnosis.csv'}")


if __name__ == "__main__":
    main()
