import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react';
import type { StudioEdge } from '../../types/nodes';

const colors = {
  raw_water: '#185abd',
  potable_water: '#3aa0f3',
  wastewater: '#8b5e3c',
  stormwater: '#00a8a8',
  reclaimed_water: '#2f9e44',
};

export function WaterEdge(props: EdgeProps<StudioEdge>) {
  const [path, labelX, labelY] = getBezierPath(props);
  const color = colors[props.data?.flowType ?? 'potable_water'];
  const markerId = `water-arrow-${props.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
  const flow = props.data?.flowMlDay;
  return (
    <>
      <defs>
        <marker
          id={markerId}
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <BaseEdge
        id={props.id}
        path={path}
        markerEnd={`url(#${markerId})`}
        style={{
          stroke: color,
          strokeWidth: props.selected ? 4 : 2.5,
          opacity: 0.9,
        }}
      />
      <EdgeLabelRenderer>
        <div
          className="edge-label nodrag nopan"
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
        >
          {flow === undefined
            ? `配比 ${((props.data?.allocation ?? 1) * 100).toFixed(0)}%`
            : `${flow.toLocaleString('zh-CN', { maximumFractionDigits: 2 })} ML/d`}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}
