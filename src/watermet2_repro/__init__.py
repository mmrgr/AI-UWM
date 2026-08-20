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
from .ai_capacity import find_ai_carrying_capacity, scan_ai_capacity
from .ai_metrics import compare_baseline_ai, infrastructure_utilization_summary, summarize_ai_water_kpis
from .data_center import calculate_data_center_plan, finalize_data_center_day
from .ai_scenarios import generate_ai_scenario_matrix, pareto_ai_strategies, run_ai_scenario_matrix
from .sensitivity import capacity_exceedance_probability, morris_sensitivity, probabilistic_ai_capacity_threshold, sobol_sensitivity

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
    "generate_ai_scenario_matrix",
    "run_ai_scenario_matrix",
    "morris_sensitivity",
    "sobol_sensitivity",
    "probabilistic_ai_capacity_threshold",
    "capacity_exceedance_probability",
    "pareto_ai_strategies",
]
