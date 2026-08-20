from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import (
    analytic_hierarchy_rank,
    compromise_programming_rank,
    evaluate_decision_problem,
    grid_calibrate,
    monte_carlo,
    pareto_grid_optimize,
)
from .full_engine import FullWaterMet2Model, load_project
from .ai_capacity import find_ai_carrying_capacity, scan_ai_capacity


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


def command_ahp_rank(args: argparse.Namespace) -> None:
    alternatives = pd.read_csv(args.alternatives)
    criteria = json.loads(Path(args.criteria).read_text(encoding="utf-8"))
    specification = json.loads(Path(args.pairwise).read_text(encoding="utf-8"))
    ranked, diagnostics = analytic_hierarchy_rank(
        alternatives,
        criteria,
        specification["matrix"],
        alternative_pairwise=specification.get("alternative_pairwise"),
        consistency_threshold=float(specification.get("consistency_threshold", 0.10)),
    )
    if specification.get("require_consistency", True) and not diagnostics.is_consistent:
        raise ValueError(
            f"AHP pairwise matrix is inconsistent (CR={diagnostics.consistency_ratio:.4f})"
        )
    if specification.get("require_consistency", True):
        inconsistent = [
            column.removeprefix("ahp_consistency_ratio_")
            for column in ranked
            if column.startswith("ahp_consistency_ratio_")
            and float(ranked[column].iloc[0])
            > float(specification.get("consistency_threshold", 0.10))
        ]
        if inconsistent:
            raise ValueError(
                f"AHP alternative pairwise matrices are inconsistent: {inconsistent}"
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    print(f"criteria consistency ratio: {diagnostics.consistency_ratio:.6f}")
    print(ranked.to_string(index=False))


def command_dss(args: argparse.Namespace) -> None:
    project, timeseries = load_project(args.project, args.timeseries)
    specification = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    result = evaluate_decision_problem(project, timeseries, specification)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.runs.to_csv(output / "scenario_strategy_runs.csv", index=False)
    result.decision_matrix.to_csv(output / "decision_matrix.csv", index=False)
    result.rankings.to_csv(output / "rankings.csv", index=False)
    print(result.rankings.to_string(index=False))


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


def command_ai_scan(args: argparse.Namespace) -> None:
    project, timeseries = load_project(args.project, args.timeseries)
    frame = scan_ai_capacity(project, timeseries, args.min_mw, args.max_mw, args.step_mw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


def command_ai_threshold(args: argparse.Namespace) -> None:
    project, timeseries = load_project(args.project, args.timeseries)
    specification = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    frame = scan_ai_capacity(
        project, timeseries,
        float(specification.get("min_mw", 0)),
        float(specification.get("max_mw", 2000)),
        float(specification.get("step_mw", 25)),
        capacities_mw=specification.get("capacities_mw"),
    )
    result = find_ai_carrying_capacity(frame, specification["constraints"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result["capacity_scan"].to_csv(output / "ai_capacity_threshold.csv", index=False)
    summary = {key: value for key, value in result.items() if key != "capacity_scan"}
    _write_json(output / "ai_capacity_threshold.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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

    ahp_rank = subparsers.add_parser("ahp-rank", help="AHP 多准则排序")
    ahp_rank.add_argument("--alternatives", required=True)
    ahp_rank.add_argument("--criteria", required=True)
    ahp_rank.add_argument("--pairwise", required=True)
    ahp_rank.add_argument("--output", default="output/ahp_ranking.csv")
    ahp_rank.set_defaults(func=command_ahp_rank)

    dss = subparsers.add_parser(
        "dss", help="批量评估场景与干预策略并按 CP/AHP 排序"
    )
    dss.add_argument("project")
    dss.add_argument("--timeseries")
    dss.add_argument("--spec", required=True)
    dss.add_argument("--output", default="output/dss")
    dss.set_defaults(func=command_dss)

    optimize = subparsers.add_parser("optimize", help="离散干预方案的多目标 Pareto 优化")
    optimize.add_argument("project")
    optimize.add_argument("--timeseries")
    optimize.add_argument("--spec", required=True)
    optimize.add_argument("--output", default="output/pareto_trials.csv")
    optimize.set_defaults(func=command_optimize)

    ai_scan = subparsers.add_parser("ai-scan", help="扫描AI容量与城市水压力")
    ai_scan.add_argument("project")
    ai_scan.add_argument("--timeseries")
    ai_scan.add_argument("--min-mw", type=float, default=0)
    ai_scan.add_argument("--max-mw", type=float, default=2000)
    ai_scan.add_argument("--step-mw", type=float, default=25)
    ai_scan.add_argument("--output", default="output/ai_capacity_scan.csv")
    ai_scan.set_defaults(func=command_ai_scan)

    ai_threshold = subparsers.add_parser("ai-threshold", help="识别AI水资源承载边界")
    ai_threshold.add_argument("project")
    ai_threshold.add_argument("--timeseries")
    ai_threshold.add_argument("--spec", required=True)
    ai_threshold.add_argument("--output", default="output/ai_threshold")
    ai_threshold.set_defaults(func=command_ai_threshold)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
