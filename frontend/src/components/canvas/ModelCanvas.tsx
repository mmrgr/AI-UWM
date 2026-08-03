import { useCallback, useEffect, useMemo, useState, type DragEvent } from 'react';
import {
  Background,
  ConnectionMode,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  useReactFlow,
  type Connection,
  type NodeChange,
} from '@xyflow/react';
import { ArrowAutofitContentRegular } from '@fluentui/react-icons';
import { isConnectionAllowed } from '../../engine/checkerRules/runDataChecker';
import { useCanvasStore } from '../../store/useCanvasStore';
import { useCheckerStore } from '../../store/useCheckerStore';
import { useProjectStore } from '../../store/useProjectStore';
import { useResultStore } from '../../store/useResultStore';
import type {
  ComponentKind,
  StudioEdge,
  StudioNode,
  WaterFlowType,
} from '../../types/nodes';
import { Button } from '../ui/Button';
import { paletteItems } from '../../data/paletteItems';
import { Palette } from './Palette';
import { WaterEdge } from './WaterEdge';
import { WaterNode } from './WaterNode';

const nodeTypes = { watermet: WaterNode };
const edgeTypes = { watermet: WaterEdge };

const layoutColumns: Record<ComponentKind, number> = {
  water_resource: 0,
  supply_conduit: 1,
  wtw: 2,
  trunk_main: 3,
  service_reservoir: 4,
  distribution_main: 5,
  local_area: 6,
  reuse: 5,
  sewer: 7,
  wwtw: 8,
  receiving_water: 9,
};

function defaultConfig(kind: ComponentKind): Record<string, unknown> {
  const base: Record<string, unknown> = { kind, name: '' };
  if (kind === 'water_resource') {
    return { ...base, capacity_ml: 1000, initial_ml: 500, abstraction_capacity_ml_day: 10 };
  }
  if (kind === 'local_area') {
    return {
      name: '',
      base_population: 1000,
      area_ha: 50,
      demand_profiles: [],
      surfaces: {
        impervious: { fraction: 0.6, runoff_coefficient: 0.8 },
        pervious: { fraction: 0.4, runoff_coefficient: 0.15 },
      },
    };
  }
  if (kind === 'receiving_water') return base;
  return { ...base, daily_capacity_ml: 10 };
}

function connectionFlow(source: ComponentKind): WaterFlowType {
  if (source === 'sewer' || source === 'wwtw') return 'wastewater';
  if (source === 'reuse') return 'reclaimed_water';
  if (source === 'water_resource' || source === 'supply_conduit') return 'raw_water';
  return 'potable_water';
}

export function ModelCanvas() {
  const nodes = useProjectStore((state) => state.nodes);
  const edges = useProjectStore((state) => state.edges);
  const onNodesChange = useProjectStore((state) => state.onNodesChange);
  const onEdgesChange = useProjectStore((state) => state.onEdgesChange);
  const connect = useProjectStore((state) => state.connect);
  const addNode = useProjectStore((state) => state.addNode);
  const diagnostics = useCheckerStore((state) => state.diagnostics);
  const selectNode = useCanvasStore((state) => state.selectNode);
  const selectEdge = useCanvasStore((state) => state.selectEdge);
  const setZoom = useCanvasStore((state) => state.setZoom);
  const focusRequest = useCanvasStore((state) => state.focusRequest);
  const addLog = useResultStore((state) => state.addLog);
  const { screenToFlowPosition, setCenter, getNode } = useReactFlow<StudioNode, StudioEdge>();
  const [connectionWarning, setConnectionWarning] = useState<string | null>(null);

  const decoratedNodes = useMemo(
    () =>
      nodes.map((node) => {
        const related = diagnostics.filter((item) => item.nodeId === node.id);
        const errorCount = related.filter((item) => item.severity === 'error').length;
        const warningCount = related.filter((item) => item.severity !== 'error').length;
        return {
          ...node,
          data: {
            ...node.data,
            errorCount,
            warningCount,
            status:
              errorCount > 0
                ? 'error'
                : warningCount > 0
                  ? 'warning'
                  : 'valid',
          },
        } satisfies StudioNode;
      }),
    [diagnostics, nodes],
  );

  useEffect(() => {
    if (!focusRequest) return;
    const node = getNode(focusRequest.nodeId);
    if (node) {
      void setCenter(
        node.position.x + (node.measured?.width ?? 170) / 2,
        node.position.y + (node.measured?.height ?? 90) / 2,
        { zoom: 1.35, duration: 500 },
      );
    }
  }, [focusRequest, getNode, setCenter]);

  const validateConnection = useCallback(
    (connection: Connection | StudioEdge): boolean => {
      const source = nodes.find((node) => node.id === connection.source);
      const target = nodes.find((node) => node.id === connection.target);
      if (!source || !target) return false;
      return isConnectionAllowed(source.data.nodeType, target.data.nodeType);
    },
    [nodes],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const source = nodes.find((node) => node.id === connection.source);
      const target = nodes.find((node) => node.id === connection.target);
      if (!source || !target) return;
      if (!isConnectionAllowed(source.data.nodeType, target.data.nodeType)) {
        const message = `${source.data.label} 不能直接连接 ${target.data.label}。`;
        setConnectionWarning(message);
        addLog(message);
        window.setTimeout(() => setConnectionWarning(null), 3500);
        return;
      }
      connect(connection, {
        flowType: connectionFlow(source.data.nodeType),
        allocation: 1,
      });
    },
    [addLog, connect, nodes],
  );

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const kind = event.dataTransfer.getData('application/watermet-node') as ComponentKind;
      const item = paletteItems.find((candidate) => candidate.nodeType === kind);
      if (!item) return;
      const prefix = {
        water_resource: 'WR',
        supply_conduit: 'SC',
        wtw: 'WTW',
        trunk_main: 'TM',
        service_reservoir: 'SR',
        distribution_main: 'DM',
        local_area: 'LA',
        reuse: 'REUSE',
        sewer: 'SEWER',
        wwtw: 'WWTW',
        receiving_water: 'RW',
      }[kind];
      let sequence = 1;
      while (nodes.some((node) => node.id === `${prefix}${sequence}`)) sequence += 1;
      const id = `${prefix}${sequence}`;
      const config = { ...defaultConfig(kind), name: `${item.label} ${sequence}` };
      addNode({
        id,
        type: 'watermet',
        position: screenToFlowPosition({ x: event.clientX, y: event.clientY }),
        data: {
          label: `${item.label} ${sequence}`,
          nodeType: kind,
          config,
          status: 'unconfigured',
          errorCount: 0,
          warningCount: 0,
        },
      });
      selectNode(id);
    },
    [addNode, nodes, screenToFlowPosition, selectNode],
  );

  const autoLayout = () => {
    const rows = new Map<number, number>();
    const changes: NodeChange<StudioNode>[] = nodes.map((node) => {
      const column = layoutColumns[node.data.nodeType];
      const row = rows.get(column) ?? 0;
      rows.set(column, row + 1);
      return {
        id: node.id,
        type: 'position',
        position: { x: 70 + column * 220, y: 80 + row * 145 },
      };
    });
    onNodesChange(changes);
  };

  return (
    <div className="model-workspace">
      <Palette />
      <div
        className="canvas-shell"
        onDrop={onDrop}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = 'move';
        }}
      >
        {connectionWarning && <div className="canvas-warning">{connectionWarning}</div>}
        <ReactFlow<StudioNode, StudioEdge>
          nodes={decoratedNodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={validateConnection}
          onNodeClick={(_, node) => selectNode(node.id)}
          onEdgeClick={(_, edge) => selectEdge(edge.id)}
          onPaneClick={() => {
            selectNode(null);
            selectEdge(null);
          }}
          onMove={(_, viewport) => setZoom(viewport.zoom)}
          connectionMode={ConnectionMode.Strict}
          snapToGrid
          snapGrid={[16, 16]}
          fitView
          minZoom={0.25}
          maxZoom={2}
          selectionOnDrag
          multiSelectionKeyCode="Control"
          deleteKeyCode={['Delete', 'Backspace']}
          defaultEdgeOptions={{ type: 'watermet', animated: true }}
        >
          <Background gap={16} size={1} color="var(--grid)" />
          <Controls position="bottom-left" />
          <MiniMap
            position="bottom-right"
            pannable
            zoomable
            nodeColor={(node) =>
              node.data?.status === 'error'
                ? '#d13438'
                : node.data?.status === 'warning'
                  ? '#f5a623'
                  : '#0f6cbd'
            }
          />
          <Panel position="top-left">
            <div className="flow-legend">
              {edges.some((edge) => edge.data?.flowMlDay !== undefined)
                ? '箭头＝流向 · 数值＝模拟期日均流量（ML/d）'
                : '箭头＝流向 · 数值＝路径分配比例'}
            </div>
          </Panel>
          <Panel position="top-right">
            <Button
              variant="secondary"
              icon={<ArrowAutofitContentRegular />}
              onClick={autoLayout}
            >
              自动布局
            </Button>
          </Panel>
        </ReactFlow>
      </div>
    </div>
  );
}
