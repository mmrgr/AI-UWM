from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from watermet2_repro.api import create_app
from watermet2_repro.full_engine import FullWaterMet2Model, load_project
from watermet2_repro.sensitivity import capacity_exceedance_probability
from watermet2_repro.cawcc import build_intraday_profile, intraday_peak_proxy, report_constraint_spec
from watermet2_repro.ai_capacity import scan_ai_capacity, summarize_constraint_boundaries
from watermet2_repro.research import (
    apply_state_pressure,
    compare_ai_industrial,
    dimensionless_margins,
    hourly_stress_scan,
    parameter_provenance,
    robustness_matrix,
    run_state_pressure_matrix,
)


def ai_fixture() -> tuple[dict, pd.DataFrame]:
    project, timeseries = load_project(Path("examples/ai_data_center/project.json"))
    project = copy.deepcopy(project)
    timeseries = timeseries.iloc[:3].copy()
    project["simulation"]["end"] = str(timeseries.date.iloc[-1].date())
    return project, timeseries


def test_period_aggregation_does_not_sum_ai_ratios() -> None:
    project, timeseries = ai_fixture()
    result = FullWaterMet2Model(project, timeseries).run()
    from watermet2_repro.full_engine import _aggregate_frame

    monthly = _aggregate_frame(
        result.data_center_daily, "MS", ["data_center_id"]
    )
    assert len(monthly) == 1
    assert 1.0 <= float(monthly.loc[0, "pue"]) <= 2.5
    assert 0.0 <= float(monthly.loc[0, "actual_reclaimed_fraction"]) <= 1.0
    assert float(monthly.loc[0, "facility_energy_mwh"]) == result.data_center_daily["facility_energy_mwh"].sum()


def test_capacity_exceedance_ignores_unresolved_thresholds() -> None:
    samples = pd.DataFrame({"maximum_safe_ai_capacity_mw": [500.0, None, 1000.0]})
    assert capacity_exceedance_probability(samples, 750.0) == 0.5


def test_api_baseline_endpoint_is_available() -> None:
    import asyncio
    import httpx

    project, timeseries = ai_fixture()
    timeseries["date"] = timeseries["date"].dt.strftime("%Y-%m-%d")
    payload = {"project": project, "timeseries": timeseries.to_dict("records")}

    async def request() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(serve_frontend=False)),
            base_url="http://test",
        ) as client:
            return await client.post("/api/ai/baseline", json=payload)

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["comparison"]


def test_api_bottleneck_and_intraday_endpoints_are_executable() -> None:
    import asyncio
    import httpx

    project, timeseries = ai_fixture()
    timeseries["date"] = timeseries["date"].dt.strftime("%Y-%m-%d")
    base = {"project": project, "timeseries": timeseries.to_dict("records")}

    async def request() -> tuple[httpx.Response, httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(serve_frontend=False)),
            base_url="http://test",
        ) as client:
            bottleneck = await client.post(
                "/api/ai/bottlenecks",
                json={
                    **base,
                    "specification": {
                        "capacities_mw": [0, 100],
                        "constraints": {
                            "system_reliability_fraction": {"operator": ">=", "value": 0.99}
                        },
                    },
                },
            )
            intraday = await client.post(
                "/api/ai/intraday-proxy",
                json={**base, "specification": {}},
            )
            return bottleneck, intraday

    bottleneck, intraday = asyncio.run(request())
    assert bottleneck.status_code == 200
    assert bottleneck.json()["constraint_boundaries"]
    assert intraday.status_code == 200
    assert len(intraday.json()["hourly"]) == len(timeseries) * 24


def test_intraday_proxy_preserves_daily_mass() -> None:
    daily = pd.DataFrame({"date": pd.to_datetime(["2030-07-01"]), "external_withdrawal_ml": [24.0]})
    hourly = intraday_peak_proxy(daily, profile=build_intraday_profile())
    assert len(hourly) == 24
    assert abs(float(hourly.external_withdrawal_ml.sum()) - 24.0) < 1e-9
    assert float(hourly.external_withdrawal_ml.max()) > float(hourly.external_withdrawal_ml.min())


def test_report_constraints_and_boundary_table() -> None:
    project, timeseries = ai_fixture()
    constraints = report_constraint_spec(project)
    assert "system_reliability_fraction" in constraints
    scan = scan_ai_capacity(project, timeseries, capacities_mw=[0, 100])
    boundaries = summarize_constraint_boundaries(scan, constraints)
    assert "constraint" in boundaries


def test_report_state_pressure_and_control_workflows() -> None:
    project, timeseries = ai_fixture()
    changed, drivers = apply_state_pressure(project, timeseries, "S2", "G3")
    assert changed["research"] == {"system_state": "S2", "pressure": "G3"}
    assert float(drivers.temperature_c.max()) > float(timeseries.temperature_c.max())
    matrix = run_state_pressure_matrix(project, timeseries, states=["S0"], pressures=["G0"])
    assert matrix.iloc[0]["system_state"] == "S0"
    comparison = compare_ai_industrial(project, timeseries)
    assert "ai" in comparison and "industrial_control" in comparison


def test_hourly_boundary_margins_robustness_and_provenance() -> None:
    project, timeseries = ai_fixture()
    constraints = {"peak_capacity_ratio": {"operator": "<=", "value": 1.0}}
    scan = hourly_stress_scan(project, timeseries, [0, 100], constraints)
    margins = dimensionless_margins(scan, constraints)
    assert len(margins) == 2
    robust = robustness_matrix(project, timeseries, [{"components.AI_DC1.load_factor": 0.8}], [0, 100], {"system_reliability_fraction": {"operator": ">=", "value": 0.99}})
    assert len(robust) == 1
    assert not parameter_provenance(project).empty
