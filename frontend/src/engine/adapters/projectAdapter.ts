import type { ComponentKind, StudioEdge, StudioNode, WaterFlowType } from '../../types/nodes';
import type { ProjectObject, WaterMetProject } from '../../types/project';

const supplyKinds = new Set<ComponentKind>([
  'water_resource',
  'supply_conduit',
  'wtw',
  'trunk_main',
  'service_reservoir',
  'distribution_main',
]);

const kindLabels: Record<ComponentKind, string> = {
  water_resource: '水资源',
  supply_conduit: '原水输水',
  wtw: '给水处理厂',
  trunk_main: '清水干管',
  service_reservoir: '服务水池',
  distribution_main: '配水管网',
  reuse: '再生水设施',
  sewer: '排水管网',
  wwtw: '污水处理厂',
  receiving_water: '受纳水体',
  local_area: '城市区域',
};

function asString(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

function componentKind(config: ProjectObject): ComponentKind {
  const kind = config.kind;
  return typeof kind === 'string' && kind in kindLabels
    ? (kind as ComponentKind)
    : 'water_resource';
}

function flowType(source: ComponentKind, target: ComponentKind): WaterFlowType {
  if (source === 'reuse') return 'reclaimed_water';
  if (source === 'sewer' || source === 'wwtw') return 'wastewater';
  if (source === 'water_resource' || target === 'wtw') return 'raw_water';
  return 'potable_water';
}

function edgeId(source: string, target: string, index: number): string {
  return `e-${source}-${target}-${index}`;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

export function projectToGraph(project: WaterMetProject): {
  nodes: StudioNode[];
  edges: StudioEdge[];
} {
  const nodes: StudioNode[] = [];
  const kinds = new Map<string, ComponentKind>();
  let index = 0;
  for (const [id, config] of Object.entries(project.components)) {
    const kind = componentKind(config);
    kinds.set(id, kind);
    nodes.push({
      id,
      type: 'watermet',
      position: { x: 80 + (index % 4) * 230, y: 80 + Math.floor(index / 4) * 170 },
      data: {
        label: asString(config.name, id),
        nodeType: kind,
        config: { ...config },
        status: 'unconfigured',
        errorCount: 0,
        warningCount: 0,
      },
    });
    index += 1;
  }
  for (const [id, config] of Object.entries(project.local_areas)) {
    kinds.set(id, 'local_area');
    nodes.push({
      id,
      type: 'watermet',
      position: { x: 80 + (index % 4) * 230, y: 80 + Math.floor(index / 4) * 170 },
      data: {
        label: asString(config.name, id),
        nodeType: 'local_area',
        config: { ...config },
        status: 'unconfigured',
        errorCount: 0,
        warningCount: 0,
      },
    });
    index += 1;
  }

  const edges: StudioEdge[] = [];
  const seen = new Set<string>();
  const add = (
    source: string,
    target: string,
    allocation: number,
    fallbackIndex: number,
  ): void => {
    if (!kinds.has(source) || !kinds.has(target)) return;
    const key = `${source}->${target}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({
      id: edgeId(source, target, fallbackIndex),
      type: 'watermet',
      source,
      target,
      animated: true,
      data: {
        flowType: flowType(kinds.get(source) ?? 'water_resource', kinds.get(target) ?? 'local_area'),
        allocation,
      },
    });
  };

  project.supply_connections?.forEach((connection, edgeIndex) => {
    add(
      asString(connection.from, ''),
      asString(connection.to, ''),
      typeof connection.allocation === 'number' ? connection.allocation : 1,
      edgeIndex,
    );
  });
  if (!project.supply_connections?.length) {
    project.supply_paths?.forEach((path, pathIndex) => {
      const chain = Array.isArray(path.chain)
        ? path.chain.filter((value): value is string => typeof value === 'string')
        : [];
      const area = asString(path.local_area, '');
      [...chain.slice(0, -1).map((source, chainIndex) => [source, chain[chainIndex + 1]]), [chain.at(-1), area]]
        .forEach(([source, target], chainIndex) => {
          if (source && target) {
            const isAreaConnection = target === area;
            add(
              source,
              target,
              isAreaConnection && typeof path.allocation === 'number' ? path.allocation : 1,
              pathIndex * 20 + chainIndex,
            );
          }
        });
    });
  }
  project.wastewater_connections?.forEach((connection, edgeIndex) => {
    add(
      asString(connection.from, ''),
      asString(connection.to, ''),
      typeof connection.fraction === 'number' ? connection.fraction : 1,
      1000 + edgeIndex,
    );
  });
  Object.entries(project.local_areas).forEach(([areaId, area], areaIndex) => {
    const sewerIds = new Set(
      ['sanitary_sewer', 'storm_sewer', 'combined_sewer']
        .map((key) => area[key])
        .filter((value): value is string => typeof value === 'string'),
    );
    [...sewerIds].forEach((sewerId, sewerIndex) => {
      add(areaId, sewerId, 1, 2000 + areaIndex * 10 + sewerIndex);
    });
  });
  Object.entries(project.components).forEach(([componentId, component], componentIndex) => {
    if (componentKind(component) === 'reuse') {
      stringList(component.source_areas).forEach((areaId, areaIndex) => {
        add(areaId, componentId, 1, 3000 + componentIndex * 20 + areaIndex);
      });
      stringList(component.target_areas).forEach((areaId, areaIndex) => {
        add(componentId, areaId, 1, 4000 + componentIndex * 20 + areaIndex);
      });
    }
    if (componentKind(component) === 'wwtw') {
      const centralReuse = component.central_reuse_component;
      if (typeof centralReuse === 'string') {
        add(componentId, centralReuse, 1, 5000 + componentIndex);
      }
    }
  });
  return { nodes, edges };
}

export function graphToProject(
  base: WaterMetProject,
  nodes: StudioNode[],
  edges: StudioEdge[],
): WaterMetProject {
  const components: Record<string, ProjectObject> = {};
  const localAreas: Record<string, ProjectObject> = {};
  const kinds = new Map(nodes.map((node) => [node.id, node.data.nodeType]));
  for (const node of nodes) {
    if (node.data.nodeType === 'local_area') {
      localAreas[node.id] = { ...node.data.config };
    } else {
      components[node.id] = { ...node.data.config, kind: node.data.nodeType };
    }
  }
  const supplyConnections: ProjectObject[] = [];
  const wastewaterConnections: ProjectObject[] = [];
  for (const edge of edges) {
    const sourceKind = kinds.get(edge.source);
    const targetKind = kinds.get(edge.target);
    const allocation = edge.data?.allocation ?? 1;
    if (
      sourceKind &&
      supplyKinds.has(sourceKind) &&
      targetKind &&
      (supplyKinds.has(targetKind) || targetKind === 'local_area')
    ) {
      supplyConnections.push({ from: edge.source, to: edge.target, allocation });
    } else if (
      (sourceKind === 'sewer' || sourceKind === 'wwtw') &&
      (targetKind === 'sewer' || targetKind === 'wwtw' || targetKind === 'receiving_water')
    ) {
      wastewaterConnections.push({ from: edge.source, to: edge.target, fraction: allocation });
    }
  }
  const supplyAdjacency = new Map<string, StudioEdge[]>();
  for (const edge of edges) {
    const sourceKind = kinds.get(edge.source);
    const targetKind = kinds.get(edge.target);
    if (
      sourceKind &&
      supplyKinds.has(sourceKind) &&
      targetKind &&
      (supplyKinds.has(targetKind) || targetKind === 'local_area')
    ) {
      supplyAdjacency.set(edge.source, [...(supplyAdjacency.get(edge.source) ?? []), edge]);
    }
  }
  const basePaths = Array.isArray(base.supply_paths) ? base.supply_paths : [];
  const supplyPaths: ProjectObject[] = [];
  const walk = (current: string, chain: string[], allocation: number, visited: Set<string>) => {
    for (const edge of supplyAdjacency.get(current) ?? []) {
      if (visited.has(edge.target)) continue;
      const edgeAllocation = edge.data?.allocation ?? 1;
      if (kinds.get(edge.target) === 'local_area') {
        const match = basePaths.find(
          (path) =>
            path.local_area === edge.target &&
            JSON.stringify(path.chain) === JSON.stringify(chain),
        );
        supplyPaths.push({
          ...match,
          id: asString(match?.id, `PATH${supplyPaths.length + 1}`),
          local_area: edge.target,
          chain,
          allocation: allocation * edgeAllocation,
          priority:
            typeof match?.priority === 'number' ? match.priority : supplyPaths.length + 1,
        });
      } else {
        walk(
          edge.target,
          [...chain, edge.target],
          allocation * edgeAllocation,
          new Set([...visited, edge.target]),
        );
      }
    }
  };
  nodes
    .filter((node) => node.data.nodeType === 'water_resource')
    .forEach((node) => walk(node.id, [node.id], 1, new Set([node.id])));
  return {
    ...base,
    components,
    local_areas: localAreas,
    supply_connections: supplyPaths.length > 0 ? [] : supplyConnections,
    supply_paths: supplyPaths,
    wastewater_connections: wastewaterConnections,
  };
}

export function displayLabel(kind: ComponentKind): string {
  return kindLabels[kind];
}
