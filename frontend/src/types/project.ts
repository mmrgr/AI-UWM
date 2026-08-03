export type Scalar = string | number | boolean | null;
export type ProjectObject = Record<string, unknown>;

export interface WaterMetProject {
  name: string;
  simulation: {
    start: string;
    end: string;
  };
  finance?: ProjectObject;
  components: Record<string, ProjectObject>;
  local_areas: Record<string, ProjectObject>;
  indoor_areas?: Record<string, ProjectObject>;
  subcatchments?: Record<string, ProjectObject>;
  supply_paths?: ProjectObject[];
  supply_connections?: ProjectObject[];
  wastewater_connections?: ProjectObject[];
  pipelines?: ProjectObject[];
  interventions?: ProjectObject[];
  [key: string]: unknown;
}

export interface TimeSeriesRow {
  date: string;
  [column: string]: Scalar;
}

export interface ColumnMapping {
  source: string;
  target: 'keep' | 'date' | 'rainfall_mm' | 'temperature_c' | 'population' | 'resource_inflow';
}
