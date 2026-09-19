from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import pandas as pd


SUPPORTED_KINDS = {
    "water_resource",
    "supply_conduit",
    "wtw",
    "trunk_main",
    "service_reservoir",
    "distribution_main",
    "sewer",
    "wwtw",
    "receiving_water",
    "reuse",
    "data_center",
}

SUPPLY_KINDS = {
    "water_resource",
    "supply_conduit",
    "wtw",
    "trunk_main",
    "service_reservoir",
    "distribution_main",
}


class ProjectValidationError(ValueError):
    """Raised when a full-model project is internally inconsistent."""


def prepare_project(project: dict[str, Any]) -> dict[str, Any]:
    """Normalize hierarchy and compile edge-based supply topology into flow paths."""
    prepared = copy.deepcopy(project)
    local_areas = prepared.get("local_areas", {})
    subcatchments = prepared.setdefault("subcatchments", {})
    for area_id, area in local_areas.items():
        subcatchment_id = area.get("subcatchment", "SYSTEM")
        subcatchment = subcatchments.setdefault(subcatchment_id, {})
        members = subcatchment.setdefault("local_areas", [])
        if area_id not in members:
            members.append(area_id)

    if prepared.get("supply_connections"):
        incoming: dict[str, list[dict[str, Any]]] = {}
        for connection in prepared["supply_connections"]:
            incoming.setdefault(connection["to"], []).append(connection)

        def upstream_paths(node: str, visiting: set[str]) -> list[tuple[list[str], float]]:
            if node in visiting:
                raise ProjectValidationError("供水拓扑包含环")
            component = prepared.get("components", {}).get(node)
            if component and component.get("kind") == "water_resource":
                return [([node], 1.0)]
            paths: list[tuple[list[str], float]] = []
            for edge in incoming.get(node, []):
                parent = edge["from"]
                allocation = float(edge.get("allocation", 1.0))
                for chain, weight in upstream_paths(parent, visiting | {node}):
                    paths.append((chain + ([node] if node in prepared["components"] else []),
                                  weight * allocation))
            return paths

        compiled = []
        for area_id in local_areas:
            for index, (chain, allocation) in enumerate(upstream_paths(area_id, set()), 1):
                compiled.append({
                    "id": f"AUTO_{area_id}_{index}",
                    "local_area": area_id,
                    "allocation": allocation,
                    "chain": chain,
                    "priority": index,
                    "demand_priority": local_areas[area_id].get("demand_priority", []),
                })
        prepared["supply_paths"] = compiled
    return prepared


def validate_project(project: dict[str, Any], timeseries: pd.DataFrame) -> None:
    project = prepare_project(project)
    errors: list[str] = []
    simulation = project.get("simulation", {})
    components = project.get("components", {})
    local_areas = project.get("local_areas", {})
    supply_paths = project.get("supply_paths", [])
    subcatchments = project.get("subcatchments", {})
    indoor_areas = project.get("indoor_areas", {})
    if project.get("demand_allocation_policy", "priority") not in {
        "priority", "resident_first", "proportional", "ai_first",
        "policy_a", "policy_b", "policy_c",
    }:
        errors.append("demand_allocation_policy 无效")

    if not simulation.get("start") or not simulation.get("end"):
        errors.append("simulation.start 和 simulation.end 必须提供")
    if not components:
        errors.append("components 不能为空")
    if not local_areas:
        errors.append("local_areas 不能为空")
    for subcatchment_id, subcatchment in subcatchments.items():
        for area_id in subcatchment.get("local_areas", []):
            if area_id not in local_areas:
                errors.append(f"subcatchment={subcatchment_id} 引用了不存在的 local_area={area_id}")
            elif local_areas[area_id].get("subcatchment", "SYSTEM") != subcatchment_id:
                errors.append(f"local_area={area_id} 的 subcatchment 归属不一致")
    for indoor_id, indoor in indoor_areas.items():
        if indoor.get("local_area") not in local_areas:
            errors.append(f"indoor_area={indoor_id} 引用了不存在的 local_area={indoor.get('local_area')}")

    for component_id, component in components.items():
        kind = component.get("kind")
        if kind not in SUPPORTED_KINDS:
            errors.append(f"组件 {component_id} 的 kind={kind!r} 不受支持")
        if kind in {"water_resource", "service_reservoir", "sewer", "wwtw", "reuse"}:
            if float(component.get("capacity_ml", 0)) < 0:
                errors.append(f"组件 {component_id} capacity_ml 不能为负")
        for field in (
            "loss_fraction",
            "leakage_fraction",
            "infiltration_fraction",
            "exfiltration_fraction",
            "export_fraction",
        ):
            if field in component and not 0 <= float(component[field]) <= 1:
                errors.append(f"组件 {component_id}.{field} 必须位于 [0,1]")
        if kind == "sewer" and component.get("capacity_mode", "storage_release") not in {
            "storage_release", "transmission"
        }:
            errors.append(f"污水组件 {component_id} capacity_mode 无效")
        if kind == "reuse" and component.get("reuse_type", "rwh") not in {
            "rwh", "gwr", "central"
        }:
            errors.append(f"回用组件 {component_id} reuse_type 无效")
        if kind == "water_resource" and component.get(
            "resource_type", "surface"
        ) not in {"surface", "groundwater", "desalination", "imported"}:
            errors.append(f"水源组件 {component_id} resource_type 无效")
        if kind == "data_center":
            if component.get("local_area") not in local_areas:
                errors.append(
                    f"数据中心 {component_id} 引用了不存在的 local_area={component.get('local_area')}"
                )
            try:
                installed_capacity = float(component.get("installed_it_capacity_mw", 0.0))
            except (TypeError, ValueError):
                installed_capacity = -1.0
            if installed_capacity < 0:
                errors.append(f"数据中心 {component_id}.installed_it_capacity_mw 必须为非负数")
            schedule_dates: list[pd.Timestamp] = []
            for entry in component.get("capacity_schedule", []):
                try:
                    schedule_date = pd.Timestamp(entry["date"])
                    schedule_dates.append(schedule_date)
                    capacity = float(entry["capacity_mw"])
                    if capacity < 0:
                        errors.append(f"数据中心 {component_id}.capacity_schedule 容量不能为负")
                except (KeyError, TypeError, ValueError):
                    errors.append(f"数据中心 {component_id}.capacity_schedule 格式无效")
            if schedule_dates and schedule_dates != sorted(schedule_dates):
                errors.append(f"数据中心 {component_id}.capacity_schedule 必须按日期升序")
            if len(schedule_dates) != len(set(schedule_dates)):
                errors.append(f"数据中心 {component_id}.capacity_schedule 日期不能重复")
            load = component.get("load", {})
            try:
                load_factor = float(component.get("load_factor", load.get("factor", 0.0)))
            except (TypeError, ValueError):
                load_factor = -1.0
            if not 0 <= load_factor <= 1:
                errors.append(f"数据中心 {component_id}.load_factor 必须位于 [0,1]")
            cooling = component.get("cooling", {})
            try:
                base_pue = float(component.get("base_pue", component.get("pue", 1.2)))
            except (TypeError, ValueError):
                base_pue = 0.0
            if base_pue < 1:
                errors.append(f"数据中心 {component_id}.base_pue 不能小于 1")
            storage_capacity = float(component.get("cooling_storage_capacity_ml", component.get("cooling_storage_ml", 0.0)))
            storage_initial = float(component.get("initial_cooling_storage_ml", 0.0))
            if storage_capacity < 0 or storage_initial < 0 or storage_initial > storage_capacity:
                errors.append(f"数据中心 {component_id} 冷却水储量配置无效")
            if cooling.get("technology", "evaporative") not in {
                "evaporative", "efficient_evaporative", "hybrid", "dry",
                "liquid_to_air", "liquid_to_water", "liquid_air", "liquid_water",
            }:
                errors.append(f"数据中心 {component_id}.cooling.technology 无效")
            try:
                cycles = float(cooling.get("cycles_of_concentration", 5.0))
            except (TypeError, ValueError):
                cycles = 0.0
            if cycles <= 1:
                errors.append(f"数据中心 {component_id} 浓缩倍数必须大于 1")
            if cooling.get("coc_mode", "fixed") not in {"fixed", "quality_limited"}:
                errors.append(f"数据中心 {component_id}.cooling.coc_mode 无效")
            for field in ("drift_fraction", "drift_fraction_of_makeup", "blowdown_return_fraction", "internal_recovery_fraction"):
                if field in cooling and not 0 <= float(cooling[field]) <= 1:
                    errors.append(f"数据中心 {component_id}.cooling.{field} 必须位于 [0,1]")
            if any(float(value) <= 0 for value in cooling.get("water_quality_limits", {}).values()):
                errors.append(f"数据中心 {component_id}.cooling.water_quality_limits 必须为正")
            source_values: list[float] = []
            for source in component.get("water_sources", {}).values():
                try:
                    fraction = float(source.get("target_fraction", 0.0))
                except (TypeError, ValueError):
                    fraction = -1.0
                source_values.append(fraction)
                if not 0 <= fraction <= 1:
                    errors.append(f"数据中心 {component_id}.water_sources.target_fraction 必须位于 [0,1]")
            source_total = sum(source_values)
            if component.get("water_sources") and abs(source_total - 1.0) > 1e-6:
                errors.append(
                    f"数据中心 {component_id}.water_sources.target_fraction 合计应为 1"
                )
            for field in ("local_connection_capacity_ml_day", "grid_connection_capacity_mw"):
                if field in component:
                    try:
                        value = float(component[field])
                    except (TypeError, ValueError):
                        value = -1.0
                    if value < 0:
                        errors.append(f"数据中心 {component_id}.{field} 必须为非负数")
            for source in component.get("water_sources", {}).values():
                for key in ("available_ml_day", "constant_available_ml_day"):
                    if key in source and float(source[key]) < 0:
                        errors.append(f"数据中心 {component_id}.water_sources.{key} 不能为负")
            load_mode = component.get("load_mode", load.get("mode", "fixed"))
            load_column = component.get("load_factor_column", load.get("timeseries_column"))
            if load_mode == "timeseries" and not load_column:
                errors.append(f"数据中心 {component_id} timeseries负荷模式必须配置列名")
            if load_mode not in {"fixed", "timeseries", "profile"}:
                errors.append(f"数据中心 {component_id}.load_mode 无效")

        for field in (
            "capital_cost_eur",
            "lifetime_years",
            "maintenance_cost_eur_year",
            "maintenance_fraction_capital_year",
        ):
            if float(component.get(field, 0.0)) < 0:
                errors.append(f"component {component_id}.{field} cannot be negative")
        if component.get("failure_model", {}).get("model", "constant") not in {
            "constant",
            "linear",
            "exponential",
            "weibull",
        }:
            errors.append(f"component {component_id}.failure_model.model is invalid")

    if project.get("supply_connections"):
        incoming_allocations: dict[str, float] = {}
        for connection in project["supply_connections"]:
            source, target = connection.get("from"), connection.get("to")
            if source not in components:
                errors.append(f"供水连接引用了不存在的 from={source}")
            if target not in components and target not in local_areas:
                errors.append(f"供水连接引用了不存在的 to={target}")
            incoming_allocations[target] = incoming_allocations.get(target, 0.0) + float(
                connection.get("allocation", 1.0)
            )
        for target, allocation in incoming_allocations.items():
            if abs(allocation - 1.0) > 1e-6:
                errors.append(f"供水节点 {target} 的入边 allocation 合计应为 1，实际为 {allocation}")

    path_names = [str(path.get("id")) for path in supply_paths]
    duplicates = [name for name, count in Counter(path_names).items() if count > 1]
    if duplicates:
        errors.append(f"supply_paths id 重复: {duplicates}")

    allocations: dict[str, float] = {area_id: 0.0 for area_id in local_areas}
    for path in supply_paths:
        area_id = path.get("local_area")
        if area_id not in local_areas:
            errors.append(f"供水路径 {path.get('id')} 指向不存在的 local_area={area_id}")
            continue
        chain = path.get("chain", [])
        if not chain:
            errors.append(f"供水路径 {path.get('id')} 的 chain 不能为空")
            continue
        for component_id in chain:
            component = components.get(component_id)
            if component is None:
                errors.append(f"供水路径 {path.get('id')} 引用了不存在的组件 {component_id}")
            elif component.get("kind") not in SUPPLY_KINDS:
                errors.append(f"供水路径 {path.get('id')} 包含非供水组件 {component_id}")
        if components.get(chain[0], {}).get("kind") != "water_resource":
            errors.append(f"供水路径 {path.get('id')} 必须以 water_resource 开始")
        if components.get(chain[-1], {}).get("kind") != "distribution_main":
            errors.append(f"供水路径 {path.get('id')} 必须以 distribution_main 结束")
        allocation = float(path.get("allocation", 0))
        if allocation < 0:
            errors.append(f"供水路径 {path.get('id')} allocation 不能为负")
        allocations[area_id] += allocation

    for area_id, total in allocations.items():
        if abs(total - 1.0) > 1e-6:
            errors.append(f"local_area={area_id} 的供水路径 allocation 合计应为 1，实际为 {total}")
        surfaces = local_areas[area_id].get("surfaces", {})
        if surfaces:
            fraction = sum(float(surface.get("fraction", 0.0)) for surface in surfaces.values())
            if abs(fraction - 1.0) > 1e-6:
                errors.append(f"local_area={area_id} 的 surfaces.fraction 合计应为 1，实际为 {fraction}")

        for surface_name, surface in surfaces.items():
            if not 0 <= float(surface.get("recharge_fraction", 1.0)) <= 1:
                errors.append(
                    f"local_area={area_id} surface={surface_name} recharge_fraction must be in [0,1]"
                )

    required_columns = {"date"}
    for component in components.values():
        if component.get("inflow_column"):
            required_columns.add(component["inflow_column"])
        if component.get("kind") == "data_center":
            load = component.get("load", {})
            if component.get("load_mode", load.get("mode")) == "timeseries":
                column = component.get("load_factor_column", load.get("timeseries_column"))
                if column:
                    required_columns.add(column)
            for source in component.get("water_sources", {}).values():
                if source.get("availability_column"):
                    required_columns.add(source["availability_column"])
    for area in local_areas.values():
        if area.get("population_column"):
            required_columns.add(area["population_column"])
        weather = area.get("weather_columns", {})
        required_columns.update(value for value in weather.values() if value)
        for profile in area.get("demand_profiles", []):
            if profile.get("column"):
                required_columns.add(profile["column"])
    for indoor in indoor_areas.values():
        if indoor.get("population_column"):
            required_columns.add(indoor["population_column"])
        for profile in indoor.get("demand_profiles", []):
            if profile.get("column"):
                required_columns.add(profile["column"])
    missing = sorted(required_columns - set(timeseries.columns))
    if missing:
        errors.append(f"timeseries 缺少列: {missing}")

    if "date" in timeseries:
        dates = pd.to_datetime(timeseries["date"], errors="coerce")
        if dates.isna().any():
            errors.append("timeseries.date 含无法解析的日期")
        elif dates.duplicated().any():
            errors.append("timeseries.date 不能重复")
        elif len(dates) > 1 and not (dates.sort_values().diff().dropna() == pd.Timedelta(days=1)).all():
            errors.append("timeseries 必须是无缺日的日序列")

    for connection in project.get("wastewater_connections", []):
        source = connection.get("from")
        target = connection.get("to")
        if source not in components and source not in local_areas:
            errors.append(f"污水连接引用了不存在的 from={source}")
        if target not in components:
            errors.append(f"污水连接引用了不存在的 to={target}")
        if float(connection.get("fraction", 0)) < 0:
            errors.append(f"污水连接 {source}->{target} fraction 不能为负")

    pipelines = {pipeline.get("id"): pipeline for pipeline in project.get("pipelines", [])}
    for pipeline_id, pipeline in pipelines.items():
        if pipeline.get("component_id") not in components:
            errors.append(f"管道 {pipeline_id} 引用了不存在的组件")
        if float(pipeline.get("length_m", 0.0)) < 0:
            errors.append(f"管道 {pipeline_id}.length_m 不能为负")
        method = pipeline.get("rehabilitation_method")
        if method and method not in project.get("rehabilitation_methods", {}):
            errors.append(f"管道 {pipeline_id} 引用了不存在的修复方法 {method}")
    for event in project.get("pipeline_events", []):
        if event.get("pipeline_id") not in pipelines:
            errors.append(f"管道事件引用了不存在的管道 {event.get('pipeline_id')}")
        if event.get("action", "rehabilitate") != "retire" and event.get(
            "method"
        ) not in project.get("rehabilitation_methods", {}):
            errors.append(f"管道事件引用了不存在的修复方法 {event.get('method')}")

    if errors:
        raise ProjectValidationError("项目配置无效：\n- " + "\n- ".join(errors))
