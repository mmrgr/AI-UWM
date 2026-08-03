from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from watermet2_repro.analysis import (
    analytic_hierarchy_rank,
    analytic_hierarchy_weights,
    evaluate_decision_problem,
)
from watermet2_repro.full_engine import FullWaterMet2Model, load_project


PROJECT = Path("examples/demo_full/project.json")


def short_project(days: int = 3):
    project, timeseries = load_project(PROJECT)
    project = copy.deepcopy(project)
    timeseries = timeseries.iloc[:days].copy()
    project["simulation"]["end"] = str(timeseries["date"].iloc[-1].date())
    project["pipeline_events"] = []
    project["interventions"] = []
    return project, timeseries


def test_risk_thresholds_use_tolerance_and_unconfigured_status() -> None:
    project, timeseries = short_project()
    project["risk_thresholds"] = {
        "R01HZ04": {
            "threshold": 0.0,
            "tolerance": 1e-9,
            "normalization_scale": 1.0,
        }
    }
    result = FullWaterMet2Model(project, timeseries).run()

    shortage = result.risk_daily[result.risk_daily.risk_code == "R01HZ04"]
    assert shortage["assessed"].all()
    assert not shortage["exceeded"].any()
    assert (shortage["tolerance"] == 1e-9).all()
    assert set(shortage["method"]) == {"direct_model_output"}

    low_pressure = result.risk_daily[result.risk_daily.risk_code == "R08HZ11"]
    assert not low_pressure["assessed"].any()
    assert not low_pressure["exceeded"].any()
    assert set(low_pressure["method"]) == {"deterministic_proxy"}
    summary = result.risk_summary.set_index("risk_code")
    assert summary.loc["R08HZ11", "assessment_status"] == "unconfigured"


def test_ahp_weights_and_consistency_for_consistent_matrix() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 4.0],
            [0.5, 1.0, 2.0],
            [0.25, 0.5, 1.0],
        ]
    )
    result = analytic_hierarchy_weights(matrix, labels=["a", "b", "c"])
    np.testing.assert_allclose(result.weights, [4 / 7, 2 / 7, 1 / 7], atol=1e-10)
    assert result.consistency_ratio == 0.0
    assert result.is_consistent


def test_ahp_rank_uses_pairwise_criteria_and_metric_goals() -> None:
    alternatives = pd.DataFrame(
        {
            "alternative": ["BAU", "Reuse", "New source"],
            "reliability": [0.80, 0.90, 0.99],
            "cost": [10.0, 12.0, 30.0],
        }
    )
    criteria = {
        "reliability": {"goal": "max"},
        "cost": {"goal": "min"},
    }
    pairwise = np.array([[1.0, 3.0], [1 / 3, 1.0]])
    ranked, diagnostics = analytic_hierarchy_rank(
        alternatives,
        criteria,
        pairwise,
    )
    assert ranked.iloc[0]["alternative"] == "New source"
    assert ranked["ahp_score"].is_monotonic_decreasing
    assert np.isclose(ranked["ahp_score"].sum(), 1.0)
    assert sorted(ranked["rank"].tolist()) == [1, 2, 3]
    assert diagnostics.is_consistent


def test_decision_problem_runs_scenarios_strategies_custom_metrics_and_preferences() -> None:
    project, timeseries = short_project(2)
    specification = {
        "scenarios": [
            {"name": "baseline"},
            {
                "name": "constrained",
                "set": [
                    {
                        "path": "components.WR1.abstraction_capacity_ml_day",
                        "value": 20.0,
                    }
                ],
            },
        ],
        "strategies": [
            {"name": "BAU"},
            {
                "name": "Leakage reduction",
                "set": [
                    {"path": "components.DM1.leakage_fraction", "value": 0.01}
                ],
            },
        ],
        "metrics": {
            "reliability_fraction": {"goal": "max"},
            "present_total_cost_eur": {"goal": "min"},
            "social_acceptance": {"goal": "max"},
        },
        "custom_indicators": [
            {
                "scenario": "*",
                "strategy": "BAU",
                "metric": "social_acceptance",
                "value": 4.0,
            },
            {
                "scenario": "*",
                "strategy": "Leakage reduction",
                "metric": "social_acceptance",
                "value": 8.0,
            },
        ],
        "preference_groups": {
            "balanced": {
                "method": "cp",
                "weights": {
                    "reliability_fraction": 0.5,
                    "present_total_cost_eur": 0.25,
                    "social_acceptance": 0.25,
                },
            },
            "consumer": {
                "method": "ahp",
                "pairwise": [
                    [1.0, 3.0, 2.0],
                    [1 / 3, 1.0, 0.5],
                    [0.5, 2.0, 1.0],
                ],
            },
        },
    }

    result = evaluate_decision_problem(project, timeseries, specification)
    assert len(result.runs) == 4
    assert len(result.decision_matrix) == 4
    assert set(result.decision_matrix["social_acceptance"]) == {4.0, 8.0}
    assert set(result.rankings["scenario"]) == {"baseline", "constrained"}
    assert set(result.rankings["preference_group"]) == {"balanced", "consumer"}
    assert set(result.rankings["method"]) == {"cp", "ahp"}
    assert result.rankings.groupby(["scenario", "preference_group"])["rank"].count().eq(2).all()
