from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .analysis import (
    evaluate_decision_problem,
    grid_calibrate,
    monte_carlo,
    pareto_grid_optimize,
    summarize_kpis,
)
from .full_engine import FullWaterMet2Model, load_project
from .validation import ProjectValidationError, prepare_project, validate_project


ROOT = Path(__file__).resolve().parents[2]
DEMO_PROJECT = ROOT / "examples" / "demo_full" / "project.json"
FRONTEND_DIST = ROOT / "frontend" / "dist"


class StudioPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: dict[str, Any]
    timeseries: list[dict[str, Any]]


class DecisionPayload(StudioPayload):
    specification: dict[str, Any]


class AnalysisPayload(StudioPayload):
    specification: dict[str, Any]


class CalibrationPayload(AnalysisPayload):
    observed: list[dict[str, Any]]


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    return clean.astype(object).where(pd.notna(clean), None).to_dict(orient="records")


def _payload_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ProjectValidationError("时间序列不能为空")
    frame = pd.DataFrame(rows)
    if "date" not in frame:
        raise ProjectValidationError("时间序列必须包含 date 列")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _run_payload(payload: StudioPayload) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    timeseries = _payload_frame(payload.timeseries)
    project = prepare_project(payload.project)
    validate_project(project, timeseries)
    return project, timeseries, FullWaterMet2Model(project, timeseries).run()


def _result_tables(result: Any) -> dict[str, list[dict[str, Any]]]:
    names = (
        "system_daily",
        "subcatchment_daily",
        "component_daily",
        "area_daily",
        "indoor_daily",
        "pollutant_daily",
        "recovery_daily",
        "material_events",
        "asset_daily",
        "flood_daily",
        "risk_daily",
        "risk_summary",
    )
    return {name: _frame_records(getattr(result, name)) for name in names}


def create_app(serve_frontend: bool = True) -> FastAPI:
    app = FastAPI(
        title="WaterMet² Studio API",
        version="1.0.0",
        description="Local API adapter for the WaterMet² Python simulation engine.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "engine": "watermet2-reproduction"}

    @app.get("/api/project/demo")
    def demo() -> dict[str, Any]:
        project, timeseries = load_project(DEMO_PROJECT)
        return {
            "project": project,
            "timeseries": _frame_records(timeseries),
        }

    @app.post("/api/validate")
    def validate(payload: StudioPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "valid": True,
            "components": len(project["components"]),
            "local_areas": len(project["local_areas"]),
            "days": len(timeseries),
        }

    @app.post("/api/run")
    def run(payload: StudioPayload) -> dict[str, Any]:
        try:
            _, _, result = _run_payload(payload)
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "summary": summarize_kpis(result),
            "tables": _result_tables(result),
        }

    @app.post("/api/dss")
    def dss(payload: DecisionPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            result = evaluate_decision_problem(
                project, timeseries, payload.specification
            )
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "runs": _frame_records(result.runs),
            "decision_matrix": _frame_records(result.decision_matrix),
            "rankings": _frame_records(result.rankings),
        }

    @app.post("/api/calibrate")
    def calibrate(payload: CalibrationPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            observed_frame = _payload_frame(payload.observed).set_index("date")
            specification = payload.specification
            observed = observed_frame[specification["observed_column"]]
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
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "trials": _frame_records(result.trials),
            "best_project": result.best_project,
        }

    @app.post("/api/uncertainty")
    def uncertainty(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            specification = payload.specification
            samples, percentiles = monte_carlo(
                project,
                timeseries,
                specification["parameters"],
                samples=int(specification.get("samples", 100)),
                seed=int(specification.get("seed", 42)),
            )
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "samples": _frame_records(samples),
            "percentiles": _frame_records(percentiles),
        }

    @app.post("/api/optimize")
    def optimize(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            specification = payload.specification
            trials = pareto_grid_optimize(
                project,
                timeseries,
                specification["decisions"],
                specification["objectives"],
            )
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"trials": _frame_records(trials)}

    @app.post("/api/export")
    def export(payload: StudioPayload) -> StreamingResponse:
        try:
            project, _, result = _run_payload(payload)
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        archive = BytesIO()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            result.write(output)
            (output / "summary.json").write_text(
                json.dumps(summarize_kpis(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output / "project.json").write_text(
                json.dumps(project, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
                for file in sorted(output.iterdir()):
                    bundle.write(file, file.name)
        archive.seek(0)
        return StreamingResponse(
            archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="watermet2-results.zip"'
            },
        )

    if serve_frontend and (FRONTEND_DIST / "index.html").is_file():
        app.mount(
            "/",
            StaticFiles(directory=FRONTEND_DIST, html=True),
            name="watermet2-studio",
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "watermet2_repro.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
