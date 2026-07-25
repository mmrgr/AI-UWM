from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import (
    compromise_programming_rank,
    grid_calibrate,
    monte_carlo,
    pareto_grid_optimize,
)
from .full_engine import FullWaterMet2Model, load_project


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def command_validate(args: argparse.Namespace) -> None:
    project, timeseries = load_project(args.project, args.timeseries)
    print(
        f"配置有效：{len(project['components'])} 个组件，"
        f"{len(project['local_areas'])} 个 Local area，{len(timeseries)} 个日步长。"
    )


def command_run(args: argparse.Namespace) -> None:
    model = FullWaterMet2Model.from_files(args.project, args.timeseries)
    result = model.run()
    result.write(args.output)
    summary = {
        "delivered_percent": float(
            100 * result.system_daily["delivered_total_ml"].sum() / result.system_daily["water_demand_ml"].sum()
        ),
        "total_unmet_ml": float(result.system_daily["unmet_demand_ml"].sum()),
        "net_ghg_kg_co2e": float(result.system_daily["ghg_net_kg_co2e"].sum()),
        "present_cost_eur": float(result.system_daily["discounted_total_cost_eur"].sum()),
    }
    _write_json(Path(args.output) / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_calibrate(args: argparse.Namespace) -> None:
    project, timeseries = load_project(args.project, args.timeseries)
    specification = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    observed_frame = pd.read_csv(args.observed, parse_dates=["date"])
    observed = observed_frame.set_index("date")[specification["observed_column"]]
    result = grid_calibrate(
        project,
        timeseries,
        observed,
        specification["parameters"],
        specification.get("result_table", "system_daily"),
        specification["result_column"],
        specification.get("selector"),
        specification.get("objective", "nse"),
        specification.get("frequency"),
        specification.get("aggregation", "sum"),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.trials.to_csv(output / "calibration_trials.csv", index=False)
    _write_json(output / "best_project.json", result.best_project)
    print(result.trials.head(10).to_string(index=False))


def command_uncertainty(args: argparse.Namespace) -> None:
    project, timeseries = load_project(args.project, args.timeseries)
    specification = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    samples, percentiles = monte_carlo(
        project,
        timeseries,
        specification["parameters"],
        samples=int(specification.get("samples", 100)),
        seed=int(specification.get("seed", 42)),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output / "uncertainty_samples.csv", index=False)
    percentiles.to_csv(output / "uncertainty_percentiles.csv", index=False)
    print(percentiles.to_string(index=False))


def command_rank(args: argparse.Namespace) -> None:
    alternatives = pd.read_csv(args.alternatives)
    criteria = json.loads(Path(args.criteria).read_text(encoding="utf-8"))
    ranked = compromise_programming_rank(alternatives, criteria, p=args.p)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    print(ranked.to_string(index=False))


def command_optimize(args: argparse.Namespace) -> None:
    project, timeseries = load_project(args.project, args.timeseries)
    specification = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    trials = pareto_grid_optimize(
        project, timeseries, specification["decisions"], specification["objectives"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output, index=False)
    print(trials.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watermet2", description="开放式 WaterMet² 全功能分析引擎")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="校验项目配置和日序列")
    validate.add_argument("project")
    validate.add_argument("--timeseries")
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="运行完整模拟")
    run.add_argument("project")
    run.add_argument("--timeseries")
    run.add_argument("--output", default="output")
    run.set_defaults(func=command_run)

    calibrate = subparsers.add_parser("calibrate", help="按 NSE/RSR/PBIAS 校准")
    calibrate.add_argument("project")
    calibrate.add_argument("--timeseries")
    calibrate.add_argument("--observed", required=True)
    calibrate.add_argument("--spec", required=True)
    calibrate.add_argument("--output", default="output/calibration")
    calibrate.set_defaults(func=command_calibrate)

    uncertainty = subparsers.add_parser("uncertainty", help="Monte Carlo 风险与不确定性分析")
    uncertainty.add_argument("project")
    uncertainty.add_argument("--timeseries")
    uncertainty.add_argument("--spec", required=True)
    uncertainty.add_argument("--output", default="output/uncertainty")
    uncertainty.set_defaults(func=command_uncertainty)

    rank = subparsers.add_parser("rank", help="Compromise Programming 多准则排序")
    rank.add_argument("--alternatives", required=True)
    rank.add_argument("--criteria", required=True)
    rank.add_argument("--p", type=float, default=2.0)
    rank.add_argument("--output", default="output/ranking.csv")
    rank.set_defaults(func=command_rank)

    optimize = subparsers.add_parser("optimize", help="离散干预方案的多目标 Pareto 优化")
    optimize.add_argument("project")
    optimize.add_argument("--timeseries")
    optimize.add_argument("--spec", required=True)
    optimize.add_argument("--output", default="output/pareto_trials.csv")
    optimize.set_defaults(func=command_optimize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
