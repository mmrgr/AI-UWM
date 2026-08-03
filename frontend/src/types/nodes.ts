import type { Edge, Node } from '@xyflow/react';
import type { ProjectObject } from './project';

export type ComponentKind =
  | 'water_resource'
  | 'supply_conduit'
  | 'wtw'
  | 'trunk_main'
  | 'service_reservoir'
  | 'distribution_main'
  | 'reuse'
  | 'sewer'
  | 'wwtw'
  | 'receiving_water'
  | 'local_area';

export type WaterFlowType =
  | 'raw_water'
  | 'potable_water'
  | 'wastewater'
  | 'stormwater'
  | 'reclaimed_water';

export type ValidationStatus = 'valid' | 'warning' | 'error' | 'unconfigured';

export interface WaterMetNodeData extends Record<string, unknown> {
  label: string;
  nodeType: ComponentKind;
  config: ProjectObject;
  status: ValidationStatus;
  errorCount: number;
  warningCount: number;
}

export interface WaterMetEdgeData extends Record<string, unknown> {
  flowType: WaterFlowType;
  allocation: number;
  flowMlDay?: number;
  flowPeriodDays?: number;
  label?: string;
}

export type StudioNode = Node<WaterMetNodeData, 'watermet'>;
export type StudioEdge = Edge<WaterMetEdgeData, 'watermet'>;

export interface PaletteItem {
  nodeType: ComponentKind;
  label: string;
  category: '供水系统' | '城市区域' | '回用系统' | '排水与污水';
  description: string;
}
