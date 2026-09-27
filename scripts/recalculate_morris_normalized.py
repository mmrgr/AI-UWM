"""Rescale existing R2 Morris outputs to their 0–1 input ranges.

The previous implementation divided each elementary effect by the native
parameter unit.  This exact algebraic rescaling requires no model rerun and
keeps the original CSVs untouched for provenance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_r2_extra import city_bounds  # noqa: E402

R2 = ROOT / "examples" / "cawcc_r2"
OUT = ROOT / "validation_artifacts" / "r2"


def convert(path: Path, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    spans = frame["parameter"].map(lambda name: bounds[name][1] - bounds[name][0])
    if spans.isna().any():
        raise ValueError(f"Missing parameter bounds in {path}")
    for field in ("mu", "mu_star", "sigma"):
        frame[field] = frame[field] * spans
    frame["input_scale"] = "0–1 parameter range"
    return frame.sort_values(["metric", "mu_star"], ascending=[True, False]).reset_index(drop=True) if "metric" in frame else frame.sort_values("mu_star", ascending=False).reset_index(drop=True)


def main() -> None:
    ai_spec = json.loads((R2 / "specs" / "sensitivity_morris.json").read_text(encoding="utf-8"))
    ai_bounds = {name: tuple(map(float, bounds)) for name, bounds in ai_spec["parameters"].items()}
    for domain in ("ha", "co"):
        folder = OUT / domain
        project = json.loads((R2 / domain / "project.json").read_text(encoding="utf-8"))
        for source, bounds in (("morris.csv", ai_bounds), ("city_morris.csv", city_bounds(project))):
            output = folder / source.replace(".csv", "_normalized.csv")
            frame = convert(folder / source, bounds)
            frame.to_csv(output, index=False, encoding="utf-8-sig")
            print(f"{domain}: {output.name}")
            print(frame[["parameter", "mu_star"]].head(7).to_string(index=False))


if __name__ == "__main__":
    main()
