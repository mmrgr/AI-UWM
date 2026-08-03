import asyncio
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import httpx

from watermet2_repro.api import create_app


app = create_app()


def request(
    method: str, path: str, json: dict[str, Any] | None = None
) -> httpx.Response:
    async def execute() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(execute())


def test_studio_health_and_demo_project() -> None:
    health = request("GET", "/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    demo = request("GET", "/api/project/demo")
    assert demo.status_code == 200
    payload = demo.json()
    assert payload["project"]["components"]
    assert len(payload["timeseries"]) == 731

    studio = request("GET", "/")
    assert studio.status_code == 200
    assert "WaterMet" in studio.text


def test_studio_validate_and_run_short_window() -> None:
    payload = request("GET", "/api/project/demo").json()
    payload["timeseries"] = payload["timeseries"][:5]
    payload["project"]["simulation"]["end"] = payload["timeseries"][-1]["date"]

    validation = request("POST", "/api/validate", json=payload)
    assert validation.status_code == 200
    assert validation.json()["days"] == 5

    response = request("POST", "/api/run", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert len(result["tables"]["system_daily"]) == 5
    assert len(result["tables"]["risk_summary"]) == 23
    assert "pollutant_daily" in result["tables"]
    assert "asset_daily" in result["tables"]
    assert "material_events" in result["tables"]
    assert result["summary"]["reliability_fraction"] > 0

    exported = request("POST", "/api/export", json=payload)
    assert exported.status_code == 200
    with ZipFile(BytesIO(exported.content)) as archive:
        names = set(archive.namelist())
    assert {
        "system_daily.csv",
        "system_weekly.csv",
        "component_daily.csv",
        "pollutant_daily.csv",
        "risk_summary.csv",
        "system_monthly.csv",
        "system_annual.csv",
        "summary.json",
        "project.json",
    } <= names


def test_studio_dss_endpoint_evaluates_cp_and_ahp() -> None:
    payload = request("GET", "/api/project/demo").json()
    payload["timeseries"] = payload["timeseries"][:2]
    payload["project"]["simulation"]["end"] = payload["timeseries"][-1]["date"]
    payload["specification"] = {
        "scenarios": [{"name": "reference"}],
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
        },
        "preference_groups": {
            "cp": {
                "method": "cp",
                "weights": {
                    "reliability_fraction": 0.7,
                    "present_total_cost_eur": 0.3,
                },
            },
            "ahp": {"method": "ahp", "pairwise": [[1.0, 2.0], [0.5, 1.0]]},
        },
    }
    response = request("POST", "/api/dss", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["runs"]) == 2
    assert len(body["decision_matrix"]) == 2
    assert len(body["rankings"]) == 4
    assert {row["method"] for row in body["rankings"]} == {"cp", "ahp"}


def test_studio_advanced_analysis_endpoints() -> None:
    payload = request("GET", "/api/project/demo").json()
    payload["timeseries"] = payload["timeseries"][:3]
    payload["project"]["simulation"]["end"] = payload["timeseries"][-1]["date"]
    simulated = request("POST", "/api/run", json=payload).json()
    observed = [
        {"date": row["date"], "observed": row["delivered_total_ml"]}
        for row in simulated["tables"]["system_daily"]
    ]

    calibration = {
        **payload,
        "observed": observed,
        "specification": {
            "observed_column": "observed",
            "parameters": {"components.DM1.leakage_fraction": [0.05, 0.1]},
            "result_table": "system_daily",
            "result_column": "delivered_total_ml",
            "objective": "rmse",
        },
    }
    response = request("POST", "/api/calibrate", json=calibration)
    assert response.status_code == 200
    assert len(response.json()["trials"]) == 2
    assert response.json()["best_project"]["components"]["DM1"]

    uncertainty = {
        **payload,
        "specification": {
            "samples": 2,
            "seed": 7,
            "parameters": [
                {
                    "path": "components.DM1.leakage_fraction",
                    "distribution": "choice",
                    "values": [0.05, 0.1],
                }
            ],
        },
    }
    response = request("POST", "/api/uncertainty", json=uncertainty)
    assert response.status_code == 200
    assert len(response.json()["samples"]) == 2
    assert response.json()["percentiles"]

    optimization = {
        **payload,
        "specification": {
            "decisions": {"components.DM1.leakage_fraction": [0.05, 0.1]},
            "objectives": {
                "present_total_cost_eur": "min",
                "reliability_fraction": "max",
            },
        },
    }
    response = request("POST", "/api/optimize", json=optimization)
    assert response.status_code == 200
    assert len(response.json()["trials"]) == 2
    assert all("is_pareto" in row for row in response.json()["trials"])
