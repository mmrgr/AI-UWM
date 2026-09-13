"""Open, auditable reproduction of the core WaterMet2 daily simulation."""

from .full_engine import FullModelResult, FullWaterMet2Model, load_project, run_full_model
from .model import generate_proxy_inputs, run_scenario
from .toolkit import WaterMet2Toolkit
from .risk import RISK_CATALOG, RiskResult, evaluate_risks
from .analysis import (
    AHPResult,
    DecisionProblemResult,
    analytic_hierarchy_rank,
    analytic_hierarchy_weights,
    evaluate_decision_problem,
)
from .validation import ProjectValidationError, validate_project
from .ai_capacity import (
    evaluate_ai_interventions,
    find_ai_carrying_capacity,
    scan_ai_capacity,
    summarize_constraint_boundaries,
)
from .ai_metrics import compare_baseline_ai, infrastructure_utilization_summary, summarize_ai_water_kpis
from .data_center import calculate_data_center_plan, finalize_data_center_day
from .ai_scenarios import generate_ai_scenario_matrix, pareto_ai_strategies, run_ai_scenario_matrix
from .sensitivity import capacity_exceedance_probability, morris_sensitivity, probabilistic_ai_capacity_threshold, sobol_sensitivity
from .cawcc import REPORT_PRESSURES, REPORT_SYSTEM_STATES, build_intraday_profile, intraday_peak_proxy, report_constraint_spec
from .research import (
    apply_state_pressure,
    compare_ai_industrial,
    dimensionless_margins,
    hourly_stress_scan,
    industrial_control_project,
    parameter_provenance,
    robustness_matrix,
    run_state_pressure_matrix,
)

__all__ = [
    "FullModelResult",
    "FullWaterMet2Model",
    "generate_proxy_inputs",
    "load_project",
    "run_full_model",
    "run_scenario",
    "ProjectValidationError",
    "validate_project",
    "WaterMet2Toolkit",
    "RISK_CATALOG",
    "RiskResult",
    "evaluate_risks",
    "AHPResult",
    "DecisionProblemResult",
    "analytic_hierarchy_rank",
    "analytic_hierarchy_weights",
    "evaluate_decision_problem",
    "calculate_data_center_plan",
    "finalize_data_center_day",
    "summarize_ai_water_kpis",
    "compare_baseline_ai",
    "infrastructure_utilization_summary",
    "scan_ai_capacity",
    "find_ai_carrying_capacity",
    "summarize_constraint_boundaries",
    "evaluate_ai_interventions",
    "generate_ai_scenario_matrix",
    "run_ai_scenario_matrix",
    "morris_sensitivity",
    "sobol_sensitivity",
    "probabilistic_ai_capacity_threshold",
    "capacity_exceedance_probability",
    "pareto_ai_strategies",
    "REPORT_SYSTEM_STATES",
    "REPORT_PRESSURES",
    "report_constraint_spec",
    "build_intraday_profile",
    "intraday_peak_proxy",
    "apply_state_pressure",
    "run_state_pressure_matrix",
    "industrial_control_project",
    "compare_ai_industrial",
    "hourly_stress_scan",
    "dimensionless_margins",
    "robustness_matrix",
    "parameter_provenance",
]
