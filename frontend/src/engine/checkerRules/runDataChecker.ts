import type { Diagnostic } from '../../types/checker';
import type { StudioEdge, StudioNode } from '../../types/nodes';
import type { TimeSeriesRow, WaterMetProject } from '../../types/project';

function numeric(config: Record<string, unknown>, key: string): number | undefined {
  const value = config[key];
  return typeof value === 'number' ? value : undefined;
}

function collectReferencedColumns(value: unknown, columns: Set<string>): void {
  if (Array.isArray(value)) {
    value.forEach((item) => collectReferencedColumns(item, columns));
    return;
  }
  if (typeof value !== 'object' || value === null) return;
  Object.entries(value).forEach(([key, item]) => {
    if (
      typeof item === 'string' &&
      (key === 'column' || key.endsWith('_column'))
    ) {
      columns.add(item);
    } else if (key === 'weather_columns' && typeof item === 'object' && item !== null) {
      Object.values(item).forEach((column) => {
        if (typeof column === 'string') columns.add(column);
      });
    } else {
      collectReferencedColumns(item, columns);
    }
  });
}

export function runDataChecker(
  project: WaterMetProject,
  nodes: StudioNode[],
  edges: StudioEdge[],
  timeseries: TimeSeriesRow[],
): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const nodeTypes = new Map(nodes.map((node) => [node.id, node.data.nodeType]));
  const incoming = new Map<string, StudioEdge[]>();
  const outgoing = new Map<string, StudioEdge[]>();
  for (const edge of edges) {
    incoming.set(edge.target, [...(incoming.get(edge.target) ?? []), edge]);
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge]);
  }
  for (const node of nodes) {
    if (!(incoming.has(node.id) || outgoing.has(node.id))) {
      diagnostics.push({
        id: `isolated-${node.id}`,
        severity: 'error',
        code: 'TOPOLOGY_ISOLATED',
        title: '孤立节点',
        message: `${node.data.label} 未连接到系统拓扑。`,
        nodeId: node.id,
      });
    }
    const capacityKinds = new Set([
      'wtw',
      'service_reservoir',
      'distribution_main',
      'sewer',
      'wwtw',
    ]);
    if (
      capacityKinds.has(node.data.nodeType) &&
      numeric(node.data.config, 'daily_capacity_ml') === undefined &&
      numeric(node.data.config, 'capacity_ml') === undefined
    ) {
      diagnostics.push({
        id: `capacity-${node.id}`,
        severity: 'missing',
        code: 'PARAM_CAPACITY',
        title: '缺少容量参数',
        message: `${node.data.label} 尚未配置容量或日处理能力。`,
        nodeId: node.id,
        field: 'daily_capacity_ml',
      });
    }
    if (node.data.nodeType === 'local_area') {
      const hasPopulation =
        typeof node.data.config.base_population === 'number' ||
        typeof node.data.config.population_column === 'string' ||
        (typeof node.data.config.number_of_properties === 'number' &&
          typeof node.data.config.occupancy_people_property === 'number');
      if (!hasPopulation) {
        diagnostics.push({
          id: `population-${node.id}`,
          severity: 'error',
          code: 'AREA_POPULATION',
          title: '区域人口未配置',
          message: `${node.data.label} 需要人口列、基准人口或房屋数×入住率。`,
          nodeId: node.id,
          field: 'base_population',
        });
      }
    }
    if (node.data.nodeType === 'data_center') {
      const capacity = numeric(node.data.config, 'installed_it_capacity_mw');
      if (capacity === undefined || capacity < 0) {
        diagnostics.push({
          id: `ai-capacity-${node.id}`, severity: 'error', code: 'AI_CAPACITY',
          title: 'AI容量未配置', message: `${node.data.label} 需要非负 installed IT capacity。`,
          nodeId: node.id, field: 'installed_it_capacity_mw',
        });
      }
      if (typeof node.data.config.local_area !== 'string') {
        diagnostics.push({
          id: `ai-area-${node.id}`, severity: 'error', code: 'AI_LOCAL_AREA',
          title: 'AI所属区域未配置', message: `${node.data.label} 需要绑定城市区域。`,
          nodeId: node.id, field: 'local_area',
        });
      }
    }
  }
  const allocationKinds = new Set([
    'water_resource',
    'supply_conduit',
    'wtw',
    'trunk_main',
    'service_reservoir',
    'distribution_main',
    'sewer',
    'wwtw',
  ]);
  for (const [source, sourceEdges] of outgoing) {
    if (!allocationKinds.has(nodeTypes.get(source) ?? '')) continue;
    const modelEdges = sourceEdges.filter((edge) => {
      const sourceKind = nodeTypes.get(source);
      const targetKind = nodeTypes.get(edge.target);
      if (sourceKind === 'sewer' || sourceKind === 'wwtw') {
        return targetKind === 'sewer' || targetKind === 'wwtw' || targetKind === 'receiving_water';
      }
      return targetKind !== 'reuse';
    });
    if (modelEdges.length <= 1) continue;
    const total = modelEdges.reduce((sum, edge) => sum + (edge.data?.allocation ?? 1), 0);
    if (Math.abs(total - 1) > 1e-6) {
      diagnostics.push({
        id: `allocation-${source}`,
        severity: 'warning',
        code: 'ALLOCATION_SUM',
        title: '分配比例不为 1',
        message: `${source} 的出边分配比例合计为 ${total.toFixed(3)}。`,
        nodeId: source,
        fix: 'normalize-allocations',
      });
    }
  }
  const resources = nodes.filter((node) => node.data.nodeType === 'water_resource');
  const distributions = nodes.filter((node) => node.data.nodeType === 'distribution_main');
  if (resources.length === 0) {
    diagnostics.push({
      id: 'missing-resource',
      severity: 'error',
      code: 'SUPPLY_START',
      title: '缺少供水起点',
      message: '至少需要一个 water_resource。',
    });
  }
  if (distributions.length === 0) {
    diagnostics.push({
      id: 'missing-distribution',
      severity: 'error',
      code: 'SUPPLY_END',
      title: '缺少配水终点',
      message: '供水链至少需要一个 distribution_main。',
    });
  }

  const requiredColumns = new Set<string>(['date']);
  collectReferencedColumns(project, requiredColumns);
  if (timeseries.length === 0) {
    diagnostics.push({
      id: 'timeseries-empty',
      severity: 'error',
      code: 'TIMESERIES_EMPTY',
      title: '时间序列为空',
      message: '请在数据中心导入覆盖模拟日期范围的 CSV。',
    });
    return diagnostics;
  }
  const availableColumns = new Set(Object.keys(timeseries[0] ?? {}));
  requiredColumns.forEach((column) => {
    if (!availableColumns.has(column)) {
      diagnostics.push({
        id: `timeseries-column-${column}`,
        severity: 'error',
        code: 'TIMESERIES_COLUMN',
        title: '缺少时间序列列',
        message: `项目配置引用了 ${column}，但 CSV 中没有该列。`,
      });
    }
  });
  const dates = timeseries.map((row) => String(row.date ?? ''));
  const validDates = dates.filter((date) => /^\d{4}-\d{2}-\d{2}$/.test(date));
  if (validDates.length !== dates.length) {
    diagnostics.push({
      id: 'timeseries-date-format',
      severity: 'error',
      code: 'TIMESERIES_DATE_FORMAT',
      title: '日期格式错误',
      message: 'date 必须使用 YYYY-MM-DD 格式，且每行都不能为空。',
    });
  }
  if (new Set(dates).size !== dates.length) {
    diagnostics.push({
      id: 'timeseries-date-duplicate',
      severity: 'error',
      code: 'TIMESERIES_DATE_DUPLICATE',
      title: '日期重复',
      message: '时间序列中存在重复日期，每天只能有一行。',
    });
  }
  if (validDates.length === dates.length) {
    const dateSet = new Set(dates);
    const start = new Date(`${project.simulation.start}T00:00:00Z`);
    const end = new Date(`${project.simulation.end}T00:00:00Z`);
    let missingDays = 0;
    for (
      let current = start;
      current <= end;
      current = new Date(current.getTime() + 86_400_000)
    ) {
      if (!dateSet.has(current.toISOString().slice(0, 10))) missingDays += 1;
    }
    if (missingDays > 0) {
      diagnostics.push({
        id: 'timeseries-date-coverage',
        severity: 'error',
        code: 'TIMESERIES_DATE_COVERAGE',
        title: '时间序列缺日',
        message: `模拟期 ${project.simulation.start} 至 ${project.simulation.end} 缺少 ${missingDays} 天数据。`,
      });
    }
  }
  requiredColumns.forEach((column) => {
    if (column === 'date' || !availableColumns.has(column)) return;
    const invalid = timeseries.filter(
      (row) => typeof row[column] !== 'number' || !Number.isFinite(row[column]),
    ).length;
    if (invalid > 0) {
      diagnostics.push({
        id: `timeseries-values-${column}`,
        severity: 'error',
        code: 'TIMESERIES_VALUE',
        title: '时间序列含无效值',
        message: `${column} 有 ${invalid} 行不是有效数值。`,
      });
    }
  });
  return diagnostics;
}

export function isConnectionAllowed(
  sourceType: StudioNode['data']['nodeType'],
  targetType: StudioNode['data']['nodeType'],
): boolean {
  const allowed: Partial<Record<StudioNode['data']['nodeType'], StudioNode['data']['nodeType'][]>> = {
    water_resource: ['supply_conduit', 'wtw'],
    supply_conduit: ['wtw'],
    wtw: ['trunk_main', 'service_reservoir'],
    trunk_main: ['service_reservoir', 'distribution_main'],
    service_reservoir: ['distribution_main'],
    distribution_main: ['local_area', 'data_center'],
    local_area: ['reuse', 'sewer', 'data_center'],
    data_center: ['sewer'],
    reuse: ['local_area', 'data_center'],
    sewer: ['wwtw', 'receiving_water'],
    wwtw: ['reuse', 'receiving_water'],
  };
  return allowed[sourceType]?.includes(targetType) ?? false;
}
