import type { Scalar } from './project';

export type ResultRow = Record<string, Scalar>;

export interface SimulationResponse {
  summary: Record<string, number>;
  tables: Record<ResultTableName, ResultRow[]>;
}

export interface DecisionSupportResponse {
  runs: ResultRow[];
  decision_matrix: ResultRow[];
  rankings: ResultRow[];
}

export interface CalibrationResponse {
  trials: ResultRow[];
  best_project: Record<string, unknown>;
}

export interface UncertaintyResponse {
  samples: ResultRow[];
  percentiles: ResultRow[];
}

export interface OptimizationResponse {
  trials: ResultRow[];
}

export type ResultTableName =
  | 'system_daily'
  | 'subcatchment_daily'
  | 'component_daily'
  | 'area_daily'
  | 'indoor_daily'
  | 'pollutant_daily'
  | 'recovery_daily'
  | 'material_events'
  | 'asset_daily'
  | 'flood_daily'
  | 'risk_daily'
  | 'risk_summary';

export type ResultFrequency = 'daily' | 'weekly' | 'monthly' | 'annual';
