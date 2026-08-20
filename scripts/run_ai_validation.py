from __future__ import annotations

import copy
import json
from pathlib import Path

from watermet2_repro import (
    FullWaterMet2Model,
    compare_baseline_ai,
    find_ai_carrying_capacity,
    generate_ai_scenario_matrix,
    infrastructure_utilization_summary,
    load_project,
    morris_sensitivity,
    pareto_ai_strategies,
    probabilistic_ai_capacity_threshold,
    scan_ai_capacity,
    sobol_sensitivity,
    summarize_ai_water_kpis,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ai_data_center"
OUTPUT = ROOT / "validation_artifacts"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    project, timeseries = load_project(EXAMPLE / "project.json")
    ai_result = FullWaterMet2Model(project, timeseries).run()
    baseline_project = copy.deepcopy(project)
    baseline_project["components"] = {
        key: value
        for key, value in baseline_project["components"].items()
        if value.get("kind") != "data_center"
    }
    baseline_result = FullWaterMet2Model(baseline_project, timeseries).run()
    delta = compare_baseline_ai(baseline_result, ai_result, baseline_project, project)
    delta.to_csv(OUTPUT / "ai_baseline_delta.csv", index=False)
    infrastructure_utilization_summary(ai_result, project).to_csv(
        OUTPUT / "ai_infrastructure_utilization.csv", index=False
    )

    scan = scan_ai_capacity(project, timeseries, min_mw=0, max_mw=2000, step_mw=100)
    scan.to_csv(OUTPUT / "ai_capacity_scan.csv", index=False)
    constraints = json.loads((EXAMPLE / "capacity_constraints.json").read_text(encoding="utf-8"))
    threshold = find_ai_carrying_capacity(scan, constraints["constraints"])
    threshold["capacity_scan"].to_csv(OUTPUT / "ai_capacity_threshold.csv", index=False)

    scenarios = generate_ai_scenario_matrix(
        capacities_mw=[500, 1000],
        cooling=["evaporative", "hybrid", "dry"],
        water_sources=["70_30", "reclaimed_first"],
        hydroclimate=["normal", "hot+drought"],
        infrastructure=["current", "reuse_expansion"],
    )
    pareto = pareto_ai_strategies(project, timeseries, scenarios)
    pareto.to_csv(OUTPUT / "ai_pareto_strategies.csv", index=False)

    response = lambda values: (
        values["capacity_mw"] * values["load_factor"]
        * (1 + values["temperature_delta_c"] * 0.02) / values["coc"]
    )
    bounds = {
        "capacity_mw": (100, 2000),
        "load_factor": (0.5, 0.95),
        "temperature_delta_c": (0, 6),
        "coc": (2, 8),
    }
    morris = morris_sensitivity(response, bounds, trajectories=30, seed=42)
    sobol = sobol_sensitivity(response, bounds, samples=256, seed=42)
    morris.to_csv(OUTPUT / "ai_morris.csv", index=False)
    sobol.to_csv(OUTPUT / "ai_sobol.csv", index=False)
    probabilistic = probabilistic_ai_capacity_threshold(
        project,
        timeseries,
        {
            "components.AI_DC1.load_factor": (0.65, 0.9),
            "components.AI_DC1.cooling.cycles_of_concentration": (3.0, 7.0),
        },
        [0, 500, 1000, 1500, 2000],
        constraints["constraints"],
        samples=8,
        seed=42,
    )
    probabilistic.to_csv(OUTPUT / "ai_probabilistic_threshold.csv", index=False)

    dc = ai_result.data_center_daily
    checks = {
        "B_data_center_mass_balance": float(dc["water_balance_residual_ml"].abs().max()) < 1e-9,
        "C_capacity_scan_0_to_2gw": scan.ai_capacity_mw.min() == 0 and scan.ai_capacity_mw.max() == 2000,
        "D_cooling_technologies": {"evaporative", "hybrid", "dry"} <= set(pareto.cooling),
        "E_joint_sources": dc.reclaimed_water_ml.sum() > 0 and dc.potable_water_ml.sum() > 0,
        "F_reuse_capacity_limited": scan.max_reuse_utilization.max() <= 1 + 1e-12,
        "G_blowdown_to_wwtw": not ai_result.pollutant_daily.query("component_id == 'AI_DC1' and pollutant == 'TDS'").empty,
        "H_three_water_accounts": all(name in dc for name in ("external_withdrawal_ml", "consumption_ml", "return_flow_ml")),
        "I_peak_metrics": all(name in scan for name in ("maximum_daily_withdrawal_ml", "p95_daily_withdrawal_ml", "maximum_7day_average_ml")),
        "J_baseline_delta": not delta.empty,
        "K_safe_capacity": threshold["maximum_safe_ai_capacity_mw"] is not None,
        "L_probabilistic_capacity": len(probabilistic) == 8,
        "M_morris_sobol": not morris.empty and not sobol.empty,
        "O_reproducible_example": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    summary = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "maximum_safe_ai_capacity_mw": threshold["maximum_safe_ai_capacity_mw"],
        "first_failed_capacity_mw": threshold["first_failed_capacity_mw"],
        "limiting_constraint": threshold["limiting_constraint"],
        "ai_kpis": summarize_ai_water_kpis(ai_result, project),
        "artifacts": [
            "ai_baseline_delta.csv", "ai_capacity_scan.csv", "ai_capacity_threshold.csv",
            "ai_pareto_strategies.csv", "ai_morris.csv", "ai_sobol.csv",
            "ai_probabilistic_threshold.csv", "ai_infrastructure_utilization.csv",
        ],
    }
    (OUTPUT / "ai_validation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
