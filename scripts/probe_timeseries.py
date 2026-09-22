"""探测单个情景 dump 出来的表结构与数据量，用于确定后续情景选取策略。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "examples" / "cawcc_r2"
sys.path.insert(0, str(ROOT / "src"))

from aiuwm.full_engine import FullAIUWMModel  # noqa: E402

TMP = ROOT / "validation_artifacts" / "_probe_ts"


def main() -> None:
    for domain in ("ha",):
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        ts = pd.read_csv(R2 / domain / "timeseries.csv")
        trial = copy.deepcopy(project)
        for cid, comp in trial["components"].items():
            if comp.get("kind") == "data_center":
                trial["components"][cid]["installed_it_capacity_mw"] = 1000.0
                trial["components"][cid].pop("capacity_schedule", None)
        result = FullAIUWMModel(trial, ts).run()
        out = TMP / domain / "1000MW"
        result.write(out)

        total_rows = 0
        total_bytes = 0
        for p in sorted(out.rglob("*.csv")):
            rows = sum(1 for _ in p.open(encoding="utf-8")) - 1
            size = p.stat().st_size
            total_rows += rows
            total_bytes += size
            if rows <= 0:
                print(f"{p.relative_to(out)!s:45s} rows={0:7d}   (empty)")
                continue
            print(f"{p.relative_to(out)!s:45s} rows={rows:7d}  {size/1024:9.1f} KB  cols={len(pd.read_csv(p, nrows=0).columns)}")
        print(f"--- {domain} 1000MW TOTAL rows={total_rows} size={total_bytes/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
