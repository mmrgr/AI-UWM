import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type NodeChange,
} from '@xyflow/react';
import { temporal } from 'zundo';
import { create } from 'zustand';
import { defaultProject, defaultTimeseries } from '../data/defaultProject';
import { graphToProject, projectToGraph } from '../engine/adapters/projectAdapter';
import type { StudioEdge, StudioNode, WaterMetEdgeData } from '../types/nodes';
import type { TimeSeriesRow, WaterMetProject } from '../types/project';
import type { SimulationResponse } from '../types/results';
import { useResultStore } from './useResultStore';

interface ProjectState {
  project: WaterMetProject;
  timeseries: TimeSeriesRow[];
  nodes: StudioNode[];
  edges: StudioEdge[];
  dirty: boolean;
  loadProject: (project: WaterMetProject, timeseries: TimeSeriesRow[]) => void;
  setTimeseries: (rows: TimeSeriesRow[]) => void;
  setSimulationDates: (start: string, end: string) => void;
  onNodesChange: (changes: NodeChange<StudioNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<StudioEdge>[]) => void;
  connect: (connection: Connection, data: WaterMetEdgeData) => void;
  addNode: (node: StudioNode) => void;
  updateNodeConfig: (id: string, patch: Record<string, unknown>) => void;
  updateNodeLabel: (id: string, label: string) => void;
  updateEdgeAllocation: (id: string, allocation: number) => void;
  normalizeOutgoing: (source: string) => void;
  applySimulationFlows: (result: SimulationResponse) => void;
  markSaved: () => void;
  syncProject: () => WaterMetProject;
}

const initialGraph = projectToGraph(defaultProject);

function withoutSimulationFlows(edges: StudioEdge[]): StudioEdge[] {
  return edges.map((edge) => {
    if (!edge.data || edge.data.flowMlDay === undefined) return edge;
    const data = { ...edge.data };
    delete data.flowMlDay;
    delete data.flowPeriodDays;
    return { ...edge, data };
  });
}

function invalidateResult(): void {
  useResultStore.getState().clear();
}

export const useProjectStore = create<ProjectState>()(
  temporal((set, get) => ({
    project: defaultProject,
    timeseries: defaultTimeseries,
    nodes: initialGraph.nodes,
    edges: initialGraph.edges,
    dirty: false,
    loadProject: (project, timeseries) => {
      const graph = projectToGraph(project);
      invalidateResult();
      set({ project, timeseries, ...graph, dirty: false });
    },
    setTimeseries: (timeseries) => {
      invalidateResult();
      set((state) => ({
        timeseries,
        edges: withoutSimulationFlows(state.edges),
        dirty: true,
      }));
    },
    setSimulationDates: (start, end) => {
      invalidateResult();
      set((state) => ({
        project: { ...state.project, simulation: { start, end } },
        edges: withoutSimulationFlows(state.edges),
        dirty: true,
      }));
    },
    onNodesChange: (changes) => {
      const changed = changes.some((change) =>
        ['add', 'remove', 'position'].includes(change.type),
      );
      if (changed) invalidateResult();
      set((state) => ({
        nodes: applyNodeChanges(changes, state.nodes),
        edges: changed ? withoutSimulationFlows(state.edges) : state.edges,
        dirty: changed ? true : state.dirty,
      }));
    },
    onEdgesChange: (changes) => {
      const changed = changes.some((change) => ['add', 'remove'].includes(change.type));
      if (changed) invalidateResult();
      set((state) => ({
        edges: applyEdgeChanges(
          changes,
          changed ? withoutSimulationFlows(state.edges) : state.edges,
        ),
        dirty: changed ? true : state.dirty,
      }));
    },
    connect: (connection, data) =>
      set((state) => {
        invalidateResult();
        return {
          edges: addEdge(
            {
              ...connection,
              id: `e-${connection.source}-${connection.target}-${Date.now()}`,
              type: 'watermet',
              animated: true,
              data,
            },
            withoutSimulationFlows(state.edges),
          ),
          dirty: true,
        };
      }),
    addNode: (node) =>
      set((state) => {
        invalidateResult();
        return {
          nodes: [...state.nodes, node],
          edges: withoutSimulationFlows(state.edges),
          dirty: true,
        };
      }),
    updateNodeConfig: (id, patch) =>
      set((state) => {
        invalidateResult();
        return {
          nodes: state.nodes.map((node) =>
            node.id === id
              ? { ...node, data: { ...node.data, config: { ...node.data.config, ...patch } } }
              : node,
          ),
          edges: withoutSimulationFlows(state.edges),
          dirty: true,
        };
      }),
    updateNodeLabel: (id, label) =>
      set((state) => {
        invalidateResult();
        return {
          nodes: state.nodes.map((node) =>
            node.id === id ? { ...node, data: { ...node.data, label } } : node,
          ),
          edges: withoutSimulationFlows(state.edges),
          dirty: true,
        };
      }),
    updateEdgeAllocation: (id, allocation) =>
      set((state) => {
        invalidateResult();
        return {
          edges: withoutSimulationFlows(state.edges).map((edge) =>
            edge.id === id && edge.data
              ? { ...edge, data: { ...edge.data, allocation } }
              : edge,
          ),
          dirty: true,
        };
      }),
    normalizeOutgoing: (source) =>
      set((state) => {
        const count = state.edges.filter((edge) => edge.source === source).length;
        if (count === 0) return state;
        invalidateResult();
        return {
          ...state,
          edges: withoutSimulationFlows(state.edges).map((edge) =>
            edge.source === source && edge.data
              ? { ...edge, data: { ...edge.data, allocation: 1 / count } }
              : edge,
          ),
          dirty: true,
        };
      }),
    applySimulationFlows: (result) =>
      set((state) => {
        const dates = new Set(
          result.tables.component_daily.map((row) => String(row.date ?? '')),
        );
        const periodDays = Math.max(1, dates.size);
        const componentOutflow = new Map<string, number>();
        const componentInflow = new Map<string, number>();
        result.tables.component_daily.forEach((row) => {
          const id = String(row.component_id ?? '');
          const outflow = typeof row.outflow_ml === 'number' ? row.outflow_ml : 0;
          const inflow = typeof row.inflow_ml === 'number' ? row.inflow_ml : 0;
          componentOutflow.set(id, (componentOutflow.get(id) ?? 0) + outflow);
          componentInflow.set(id, (componentInflow.get(id) ?? 0) + inflow);
        });
        const areaWastewater = new Map<string, number>();
        result.tables.area_daily.forEach((row) => {
          const id = String(row.area_id ?? '');
          const sanitary =
            typeof row.sanitary_sewage_ml === 'number' ? row.sanitary_sewage_ml : 0;
          const runoff =
            typeof row.runoff_to_sewer_ml === 'number' ? row.runoff_to_sewer_ml : 0;
          areaWastewater.set(id, (areaWastewater.get(id) ?? 0) + sanitary + runoff);
        });
        const nodeKinds = new Map(state.nodes.map((node) => [node.id, node.data.nodeType]));
        const reuseInputCounts = new Map<string, number>();
        state.edges.forEach((edge) => {
          if (
            nodeKinds.get(edge.source) === 'local_area' &&
            nodeKinds.get(edge.target) === 'reuse'
          ) {
            reuseInputCounts.set(edge.target, (reuseInputCounts.get(edge.target) ?? 0) + 1);
          }
        });
        return {
          edges: state.edges.map((edge) => {
            const sourceKind = nodeKinds.get(edge.source);
            const targetKind = nodeKinds.get(edge.target);
            let total = componentOutflow.get(edge.source) ?? 0;
            if (sourceKind === 'local_area' && targetKind === 'sewer') {
              total = areaWastewater.get(edge.source) ?? 0;
            } else if (targetKind === 'reuse' && sourceKind !== 'wwtw') {
              total =
                (componentInflow.get(edge.target) ?? 0) /
                Math.max(1, reuseInputCounts.get(edge.target) ?? 1);
            } else if (targetKind === 'reuse') {
              total = componentInflow.get(edge.target) ?? 0;
            } else {
              total *= edge.data?.allocation ?? 1;
            }
            return edge.data
              ? {
                  ...edge,
                  data: {
                    ...edge.data,
                    flowMlDay: total / periodDays,
                    flowPeriodDays: periodDays,
                  },
                }
              : edge;
          }),
        };
      }),
    markSaved: () => set({ dirty: false }),
    syncProject: () => {
      const state = get();
      const project = graphToProject(state.project, state.nodes, state.edges);
      set({ project });
      return project;
    },
  })),
);
