"""Build a multi-sequence synthetic hydrology ensemble for the reservoir gate.

The published 10-year result (``reservoir_multiyear_10y.json``) rests on one
deterministic synthetic sequence.  A single sequence cannot tell a reader
whether the joint gate is a property of the endpoint or an artefact of that
particular draw.  This script repeats the same gate on many independent
sequences (different driver seeds, identical endpoint rules and capacity grid)
and reports the distribution of gate capacity.

This is still synthetic evidence, not measured hydrology.  Its purpose is to
test the *stability of the qualitative claim* (the reservoir gate sits far below
the facility-throughput threshold), which is exactly what T0.4 in the R5 plan
asks for -- not to produce a defensible absolute capacity.
"""
from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "src"), str(ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import audit_reservoir_multiyear as multiyear  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="ha,co")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--max-mw", type=float, default=2000.0)
    parser.add_argument("--step-mw", type=float, default=50.0)
    parser.add_argument("--sequences", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20260918)
    parser.add_argument("--stride", type=int, default=100003,
                        help="质数步长，避免相邻种子序列相关")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ensemble: dict[str, dict] = {}
    for domain in args.domains.split(","):
        domain = domain.strip()
        gates: list[float | None] = []
        terminal100: list[float | None] = []
        per_sequence: list[dict] = []
        for index in range(args.sequences):
            seed = args.base_seed + index * args.stride
            report = multiyear.audit_domain(
                domain, args.years, args.max_mw, args.step_mw, seed
            )
            gates.append(report["last_joint_gate_mw"])
            terminal100.append(report["last_terminal_ge_100pct_mw"])
            per_sequence.append(
                {
                    "sequence_index": index,
                    "seed": seed,
                    "last_joint_gate_mw": report["last_joint_gate_mw"],
                    "first_joint_gate_failure_mw": report["first_joint_gate_failure_mw"],
                    "last_terminal_ge_100pct_mw": report["last_terminal_ge_100pct_mw"],
                }
            )
            print(
                f"[{domain}] seq {index + 1}/{args.sequences} seed={seed} "
                f"joint_gate={report['last_joint_gate_mw']} "
                f"terminal100={report['last_terminal_ge_100pct_mw']}",
                flush=True,
            )

        numeric = [float(v) for v in gates if v is not None]
        ensemble[domain] = {
            "domain": domain,
            "years": args.years,
            "sequences": args.sequences,
            "base_seed": args.base_seed,
            "stride": args.stride,
            "hydrology_basis": (
                "independent synthetic sequences from build_r2_domains driver rules; "
                "same endpoint parameters, different driver seed"
            ),
            "joint_gate_mw_values": gates,
            "last_terminal_ge_100pct_mw_values": terminal100,
            "joint_gate_mw_min": min(numeric) if numeric else None,
            "joint_gate_mw_max": max(numeric) if numeric else None,
            "joint_gate_mw_median": statistics.median(numeric) if numeric else None,
            "joint_gate_mw_stdev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
            "censored_sequences": sum(1 for v in gates if v is None),
            "per_sequence": per_sequence,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ensemble, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written -> {out}")
    for domain, report in ensemble.items():
        print(
            f"{domain}: joint gate 中位数={report['joint_gate_mw_median']} "
            f"范围=[{report['joint_gate_mw_min']}, {report['joint_gate_mw_max']}] "
            f"std={report['joint_gate_mw_stdev']:.1f} "
            f"删失={report['censored_sequences']}/{report['sequences']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
