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
]
