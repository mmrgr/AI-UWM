import { beforeEach, describe, expect, it } from 'vitest';
import { defaultProject, defaultTimeseries } from '../data/defaultProject';
import type { ResultTableName, SimulationResponse } from '../types/results';
import { useProjectStore } from './useProjectStore';

const emptyTables = Object.fromEntries(
  [
    'system_daily',
    'subcatchment_daily',
    'component_daily',
    'area_daily',
    'indoor_daily',
    'pollutant_daily',
    'recovery_daily',
    'material_events',
    'asset_daily',
    'flood_daily',
    'risk_daily',
    'risk_summary',
  ].map((name) => [name, []]),
) as Record<ResultTableName, []>;

describe('simulation flow overlay', () => {
  beforeEach(() => {
    useProjectStore.getState().loadProject(defaultProject, defaultTimeseries);
  });

  it('converts component totals to mean edge flow in ML/day', () => {
    const result: SimulationResponse = {
      summary: {},
      tables: {
        ...emptyTables,
        component_daily: [
          { date: '2025-01-01', component_id: 'WR1', outflow_ml: 10, inflow_ml: 12 },
          { date: '2025-01-02', component_id: 'WR1', outflow_ml: 20, inflow_ml: 22 },
        ],
        area_daily: [
          {
            date: '2025-01-01',
            area_id: 'LA1',
            sanitary_sewage_ml: 5,
            runoff_to_sewer_ml: 2,
          },
          {
            date: '2025-01-02',
            area_id: 'LA1',
            sanitary_sewage_ml: 7,
            runoff_to_sewer_ml: 3,
          },
        ],
      },
    };
    useProjectStore.getState().applySimulationFlows(result);
    const supplyEdge = useProjectStore
      .getState()
      .edges.find((edge) => edge.source === 'WR1' && edge.target === 'WTW1');
    const sewerEdge = useProjectStore
      .getState()
      .edges.find((edge) => edge.source === 'LA1' && edge.target === 'SEWER1');
    expect(supplyEdge?.data?.flowMlDay).toBe(15);
    expect(sewerEdge?.data?.flowMlDay).toBe(8.5);

    useProjectStore.getState().setTimeseries(defaultTimeseries);
    const cleared = useProjectStore
      .getState()
      .edges.find((edge) => edge.source === 'WR1' && edge.target === 'WTW1');
    expect(cleared?.data?.flowMlDay).toBeUndefined();
  });
});
