import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useResultStore } from '../../store/useResultStore';
import type { StudioNode } from '../../types/nodes';
import { displayLabel } from '../../engine/adapters/projectAdapter';

function formatValue(value: unknown): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  if (Math.abs(value) >= 1000) return value.toLocaleString('zh-CN', { maximumFractionDigits: 0 });
  return value.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
}

export function WaterNode({ id, data, selected }: NodeProps<StudioNode>) {
  const result = useResultStore((state) => state.result);
  const rows = result?.tables.component_daily.filter((row) => row.component_id === id) ?? [];
  const latest = rows.at(-1);
  const metric = formatValue(latest?.outflow_ml ?? latest?.storage_ml);
  return (
    <div
      className={[
        'water-node',
        `status-${data.status}`,
        `kind-${data.nodeType}`,
        selected ? 'selected' : '',
      ].join(' ')}
    >
      <Handle id="target-left" type="target" position={Position.Left} className="port port-input" />
      <Handle id="target-top" type="target" position={Position.Top} className="port port-input" />
      <div className="node-icon">{data.nodeType === 'local_area' ? '▦' : '◈'}</div>
      <div className="node-content">
        <strong>{data.label}</strong>
        <span>{displayLabel(data.nodeType)}</span>
      </div>
      {(data.errorCount > 0 || data.warningCount > 0) && (
        <div className="node-issues">
          {data.errorCount > 0 && <b>{data.errorCount}</b>}
          {data.warningCount > 0 && <i>{data.warningCount}</i>}
        </div>
      )}
      {metric && <div className="node-result">{metric} ML/d</div>}
      <Handle id="source-right" type="source" position={Position.Right} className="port port-output" />
      <Handle id="source-bottom" type="source" position={Position.Bottom} className="port port-output" />
    </div>
  );
}
