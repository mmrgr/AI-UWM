from __future__ import annotations

import json
import copy
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
from .ai_capacity import (
    evaluate_ai_interventions,
    find_ai_carrying_capacity,
    scan_ai_capacity,
    summarize_constraint_boundaries,
)
from .ai_metrics import compare_baseline_ai, summarize_ai_water_kpis
from .ai_scenarios import run_ai_scenario_matrix
from .cawcc import build_intraday_profile, intraday_peak_proxy
from .research import (
    compare_ai_industrial,
    dimensionless_margins,
    hourly_stress_scan,
    parameter_provenance,
    robustness_matrix,
    run_state_pressure_matrix,
)
from .sensitivity import (
    capacity_exceedance_probability,
    morris_sensitivity,
    probabilistic_ai_capacity_threshold,
    sobol_sensitivity,
)
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
        "data_center_daily",
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
            project, _, result = _run_payload(payload)
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "summary": summarize_kpis(result),
            "ai_water_summary": summarize_ai_water_kpis(result, project),
            "tables": _result_tables(result),
        }

    @app.post("/api/ai/capacity-scan")
    def ai_capacity_scan(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            specification = payload.specification
            frame = scan_ai_capacity(
                project, timeseries,
                min_mw=float(specification.get("min_mw", 0)),
                max_mw=float(specification.get("max_mw", 2000)),
                step_mw=float(specification.get("step_mw", 25)),
                capacities_mw=specification.get("capacities_mw"),
                data_center_id=specification.get("data_center_id"),
            )
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"capacity_scan": _frame_records(frame)}

    @app.post("/api/ai/carrying-capacity")
    def ai_carrying_capacity(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            specification = payload.specification
            frame = scan_ai_capacity(
                project, timeseries,
                capacities_mw=specification.get("capacities_mw"),
                min_mw=float(specification.get("min_mw", 0)),
                max_mw=float(specification.get("max_mw", 2000)),
                step_mw=float(specification.get("step_mw", 25)),
            )
            threshold = find_ai_carrying_capacity(frame, specification["constraints"])
            threshold["capacity_scan"] = _frame_records(threshold["capacity_scan"])
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return threshold

    @app.post("/api/ai/bottlenecks")
    def ai_bottlenecks(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            specification = payload.specification
            scan = scan_ai_capacity(
                project,
                timeseries,
                min_mw=float(specification.get("min_mw", 0)),
                max_mw=float(specification.get("max_mw", 2000)),
                step_mw=float(specification.get("step_mw", 25)),
                capacities_mw=specification.get("capacities_mw"),
            )
            boundaries = summarize_constraint_boundaries(scan, specification["constraints"])
            interventions = evaluate_ai_interventions(
                project,
                timeseries,
                specification.get("interventions", []),
                specification["constraints"],
                capacities_mw=specification.get("capacities_mw") or scan["ai_capacity_mw"].tolist(),
            )
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "constraint_boundaries": _frame_records(boundaries),
            "interventions": _frame_records(interventions),
        }

    @app.post("/api/ai/intraday-proxy")
    def ai_intraday_proxy(payload: AnalysisPayload) -> dict[str, Any]:
        """Return a mass-preserving hourly stress proxy from daily AI output."""
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            result = FullWaterMet2Model(project, timeseries).run()
            daily = result.data_center_daily.groupby("date", as_index=False).sum(numeric_only=True)
            specification = payload.specification or {}
            profile = build_intraday_profile(
                training_fraction=float(specification.get("training_fraction", 0.65)),
                inference_fraction=float(specification.get("inference_fraction", 0.35)),
                inference_peak_factor=float(specification.get("inference_peak_factor", 1.35)),
                peak_hours=specification.get("peak_hours", list(range(10, 18))),
            )
            hourly = intraday_peak_proxy(daily, profile=profile)
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"profile": _frame_records(profile), "hourly": _frame_records(hourly)}

    @app.post("/api/ai/hourly-boundary")
    def ai_hourly_boundary(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            spec = payload.specification
            frame = hourly_stress_scan(
                project, timeseries, spec.get("capacities_mw", []), spec["constraints"],
                profile=build_intraday_profile(
                    training_fraction=float(spec.get("training_fraction", 0.65)),
                    inference_fraction=float(spec.get("inference_fraction", 0.35)),
                    inference_peak_factor=float(spec.get("inference_peak_factor", 1.35)),
                    peak_hours=spec.get("peak_hours", list(range(10, 18))),
                ),
                city_phase_hours=int(spec.get("city_phase_hours", 0)),
            )
            margins = dimensionless_margins(frame, spec["constraints"])
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"scan": _frame_records(frame), "margins": _frame_records(margins)}

    @app.post("/api/ai/states")
    def ai_states(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            spec = payload.specification or {}
            frame = run_state_pressure_matrix(project, timeseries, spec.get("states", ("S0", "S1", "S2", "S3")), spec.get("pressures", ("G0", "G1", "G2", "G3")))
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"scenarios": _frame_records(frame)}

    @app.post("/api/ai/industrial-control")
    def ai_industrial_control(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            frame = compare_ai_industrial(project, timeseries, str((payload.specification or {}).get("mode", "annual_water")))
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"comparison": _frame_records(frame)}

    @app.post("/api/ai/robustness")
    def ai_robustness(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            spec = payload.specification
            frame = robustness_matrix(project, timeseries, spec["parameter_sets"], spec["capacities_mw"], spec["constraints"])
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"samples": _frame_records(frame)}

    @app.post("/api/ai/provenance")
    def ai_provenance(payload: StudioPayload) -> dict[str, Any]:
        return {"parameters": _frame_records(parameter_provenance(payload.project))}

    @app.post("/api/ai/scenarios")
    def ai_scenarios(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            frame = run_ai_scenario_matrix(project, timeseries, payload.specification["scenarios"])
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"scenarios": _frame_records(frame)}

    @app.post("/api/ai/baseline")
    def ai_baseline(payload: StudioPayload) -> dict[str, Any]:
        """Compare the submitted city with an identical no-data-centre baseline."""
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            ai_project = copy.deepcopy(project)
            ai_ids = [
                component_id
                for component_id, component in ai_project["components"].items()
                if component.get("kind") == "data_center"
            ]
            if not ai_ids:
                raise ValueError("Project contains no data_center component")
            ai_result = FullWaterMet2Model(ai_project, timeseries).run()
            baseline_project = copy.deepcopy(ai_project)
            baseline_project["components"] = {
                component_id: component
                for component_id, component in baseline_project["components"].items()
                if component_id not in ai_ids
            }
            baseline_result = FullWaterMet2Model(baseline_project, timeseries).run()
            comparison = compare_baseline_ai(
                baseline_result, ai_result, baseline_project, ai_project
            )
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"comparison": _frame_records(comparison)}

    @app.post("/api/ai/sensitivity")
    def ai_sensitivity(payload: AnalysisPayload) -> dict[str, Any]:
        """Run Morris or Sobol sensitivity on any AI-water KPI."""
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            specification = payload.specification
            bounds = {
                str(path): (float(values[0]), float(values[1]))
                for path, values in specification["parameters"].items()
            }
            if not bounds or any(low > high for low, high in bounds.values()):
                raise ValueError("sensitivity parameters require [low, high] bounds")
            metric = str(specification.get("metric", "total_withdrawal_ml"))

            def evaluator(values: dict[str, float]) -> float:
                trial = copy.deepcopy(project)
                for path, value in values.items():
                    target: Any = trial
                    parts = path.split(".")
                    for part in parts[:-1]:
                        target = target[int(part)] if isinstance(target, list) else target[part]
                    if isinstance(target, list):
                        target[int(parts[-1])] = value
                    else:
                        target[parts[-1]] = value
                result = FullWaterMet2Model(trial, timeseries).run()
                kpis = summarize_ai_water_kpis(result, trial)
                if metric in kpis:
                    return float(kpis[metric])
                raise KeyError(f"Unknown AI KPI: {metric}")

            method = str(specification.get("method", "morris")).lower()
            if method == "morris":
                frame = morris_sensitivity(
                    evaluator, bounds,
                    trajectories=int(specification.get("trajectories", 20)),
                    seed=int(specification.get("seed", 42)),
                )
            elif method == "sobol":
                frame = sobol_sensitivity(
                    evaluator, bounds,
                    samples=int(specification.get("samples", 256)),
                    seed=int(specification.get("seed", 42)),
                )
            else:
                raise ValueError("sensitivity method must be morris or sobol")
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"method": method, "metric": metric, "sensitivity": _frame_records(frame)}

    @app.post("/api/ai/probabilistic-threshold")
    def ai_probabilistic_threshold(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            timeseries = _payload_frame(payload.timeseries)
            project = prepare_project(payload.project)
            validate_project(project, timeseries)
            specification = payload.specification
            parameter_ranges = {
                str(path): (float(values[0]), float(values[1]))
                for path, values in specification["parameter_ranges"].items()
            }
            frame = probabilistic_ai_capacity_threshold(
                project, timeseries, parameter_ranges,
                [float(value) for value in specification["capacities_mw"]],
                specification["constraints"],
                samples=int(specification.get("samples", 100)),
                seed=int(specification.get("seed", 42)),
            )
        except (ProjectValidationError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"threshold_samples": _frame_records(frame)}

    @app.post("/api/ai/exceedance-probability")
    def ai_exceedance_probability(payload: AnalysisPayload) -> dict[str, Any]:
        try:
            samples = pd.DataFrame(payload.specification["threshold_samples"])
            probability = capacity_exceedance_probability(
                samples, float(payload.specification["proposed_capacity_mw"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "proposed_capacity_mw": float(payload.specification["proposed_capacity_mw"]),
            "exceedance_probability": None if np.isnan(probability) else probability,
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
