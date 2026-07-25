from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .full_engine import FullModelResult, FullWaterMet2Model, load_project
from .validation import prepare_project, validate_project


def _resolve(root: Any, path: str) -> tuple[Any, str]:
    parts = path.split(".")
    target = root
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    return target, parts[-1]


class WaterMet2Toolkit:
    """Load/set/run/retrieve API replacing the unavailable proprietary DLL Toolkit."""

    def __init__(self, project: dict[str, Any], timeseries: pd.DataFrame):
        self.project = prepare_project(project)
        self.timeseries = timeseries.copy()
        self.result: FullModelResult | None = None

    @classmethod
    def open(
        cls, project_path: str | Path, timeseries_path: str | Path | None = None
    ) -> "WaterMet2Toolkit":
        project, timeseries = load_project(project_path, timeseries_path)
        return cls(project, timeseries)

    def clone(self) -> "WaterMet2Toolkit":
        return WaterMet2Toolkit(copy.deepcopy(self.project), self.timeseries.copy())

    def save_project(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.project, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get_input(self, path: str) -> Any:
        target: Any = self.project
        for part in path.split("."):
            target = target[int(part)] if isinstance(target, list) else target[part]
        return copy.deepcopy(target)

    def set_input(self, path: str, value: Any) -> None:
        target, key = _resolve(self.project, path)
        if isinstance(target, list):
            target[int(key)] = value
        else:
            target[key] = value
        self.result = None

    def validate(self) -> None:
        validate_project(self.project, self.timeseries)

    def run(self) -> FullModelResult:
        self.validate()
        self.result = FullWaterMet2Model(self.project, self.timeseries).run()
        return self.result

    def list_components(self, kind: str | None = None) -> list[str]:
        return [
            component_id
            for component_id, component in self.project["components"].items()
            if kind is None or component["kind"] == kind
        ]

    def list_result_tables(self) -> list[str]:
        return list(FullModelResult.__dataclass_fields__)

    def get_result(
        self,
        table: str,
        *,
        component_id: str | None = None,
        area_id: str | None = None,
        subcatchment_id: str | None = None,
        indoor_id: str | None = None,
        risk_code: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("尚未运行模型")
        frame = getattr(self.result, table).copy()
        filters = {
            "component_id": component_id,
            "area_id": area_id,
            "subcatchment_id": subcatchment_id,
            "indoor_id": indoor_id,
            "risk_code": risk_code,
        }
        for column, value in filters.items():
            if value is not None:
                if column not in frame:
                    raise KeyError(f"结果表 {table} 不含筛选字段 {column}")
                frame = frame[frame[column].astype(str) == str(value)]
        return frame[columns].copy() if columns else frame

    def write_results(self, output_dir: str | Path) -> None:
        if self.result is None:
            raise RuntimeError("尚未运行模型")
        self.result.write(output_dir)
