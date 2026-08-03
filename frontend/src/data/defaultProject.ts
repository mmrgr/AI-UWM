import type { TimeSeriesRow, WaterMetProject } from '../types/project';

export const defaultProject: WaterMetProject = {
  name: 'WaterMet² Studio 示例项目',
  simulation: { start: '2025-01-01', end: '2025-01-30' },
  finance: { base_date: '2025-01-01', discount_rate: 0.03 },
  components: {
    WR1: {
      kind: 'water_resource',
      capacity_ml: 12000,
      initial_ml: 6000,
      inflow_column: 'source_inflow_ml',
      abstraction_capacity_ml_day: 80,
    },
    WTW1: {
      kind: 'wtw',
      daily_capacity_ml: 80,
      loss_fraction: 0.01,
      electricity_kwh_m3: 0.34,
    },
    SR1: {
      kind: 'service_reservoir',
      capacity_ml: 300,
      initial_ml: 150,
      daily_capacity_ml: 100,
    },
    DM1: {
      kind: 'distribution_main',
      daily_capacity_ml: 75,
      leakage_fraction: 0.12,
      electricity_kwh_m3: 0.2,
    },
    SEWER1: {
      kind: 'sewer',
      sewer_type: 'combined',
      capacity_mode: 'transmission',
      capacity_ml: 100,
      daily_capacity_ml: 100,
      overflow_to: 'RW1',
    },
    WWTW1: {
      kind: 'wwtw',
      capacity_ml: 50,
      initial_ml: 0,
      daily_capacity_ml: 100,
      overflow_to: 'RW1',
      pollutant_removal_fraction: { BOD: 0.9, TSS: 0.92 },
    },
    RW1: { kind: 'receiving_water' },
  },
  local_areas: {
    LA1: {
      subcatchment: 'SUB1',
      base_population: 20000,
      area_ha: 400,
      weather_columns: {
        precipitation: 'rainfall_mm',
        temperature: 'temperature_c',
      },
      surfaces: {
        roof: { fraction: 0.3, runoff_coefficient: 0.85 },
        road: { fraction: 0.3, runoff_coefficient: 0.8 },
        pervious: { fraction: 0.4, runoff_coefficient: 0.15 },
      },
      demand_profiles: [
        {
          name: 'domestic',
          base_value: 145,
          unit: 'l_capita_day',
          return_fraction: 0.95,
        },
      ],
      combined_sewer: 'SEWER1',
      sanitary_pollutant_load_kg_capita_day: { BOD: 0.06, TSS: 0.07 },
    },
  },
  subcatchments: { SUB1: { local_areas: ['LA1'] } },
  supply_connections: [
    { from: 'WR1', to: 'WTW1', allocation: 1 },
    { from: 'WTW1', to: 'SR1', allocation: 1 },
    { from: 'SR1', to: 'DM1', allocation: 1 },
    { from: 'DM1', to: 'LA1', allocation: 1 },
  ],
  wastewater_connections: [
    { from: 'SEWER1', to: 'WWTW1', fraction: 1 },
    { from: 'WWTW1', to: 'RW1', fraction: 1 },
  ],
};

export const defaultTimeseries: TimeSeriesRow[] = Array.from(
  { length: 30 },
  (_, index) => {
    const date = new Date(Date.UTC(2025, 0, index + 1));
    return {
      date: date.toISOString().slice(0, 10),
      rainfall_mm: index % 6 === 0 ? 12 + (index % 5) : 0,
      temperature_c: 4 + 5 * Math.sin(index / 8),
      source_inflow_ml: 45 + 6 * Math.cos(index / 5),
    };
  },
);
