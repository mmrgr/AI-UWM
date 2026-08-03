import { describe, expect, it } from 'vitest';
import { defaultProject, defaultTimeseries } from '../../data/defaultProject';
import { projectToGraph } from '../adapters/projectAdapter';
import { isConnectionAllowed, runDataChecker } from './runDataChecker';

describe('runDataChecker', () => {
  it('accepts the built-in connected model without structural errors', () => {
    const graph = projectToGraph(defaultProject);
    const diagnostics = runDataChecker(
      defaultProject,
      graph.nodes,
      graph.edges,
      defaultTimeseries,
    );
    expect(diagnostics.filter((item) => item.severity === 'error')).toEqual([]);
  });

  it('detects isolation and invalid allocation totals', () => {
    const graph = projectToGraph(defaultProject);
    const isolated = {
      ...graph.nodes[0]!,
      id: 'ISOLATED',
      data: { ...graph.nodes[0]!.data, label: '孤立节点' },
    };
    const split = graph.edges.map((edge, index) =>
      index === 0 && edge.data
        ? { ...edge, data: { ...edge.data, allocation: 0.4 } }
        : edge,
    );
    const diagnostics = runDataChecker(
      defaultProject,
      [...graph.nodes, isolated],
      split,
      defaultTimeseries,
    );
    expect(diagnostics.some((item) => item.code === 'TOPOLOGY_ISOLATED')).toBe(true);
  });

  it('blocks physically invalid direct connections', () => {
    expect(isConnectionAllowed('water_resource', 'wtw')).toBe(true);
    expect(isConnectionAllowed('wtw', 'wwtw')).toBe(false);
  });

  it('detects missing columns, duplicate dates, and incomplete coverage', () => {
    const graph = projectToGraph(defaultProject);
    const rows = defaultTimeseries.slice(0, 2).map((row) => ({
      date: row.date,
      rainfall_mm: row.rainfall_mm ?? null,
      temperature_c: row.temperature_c ?? null,
    }));
    rows[1] = { ...rows[1]!, date: rows[0]!.date };
    const diagnostics = runDataChecker(defaultProject, graph.nodes, graph.edges, rows);
    expect(diagnostics.map((item) => item.code)).toEqual(
      expect.arrayContaining([
        'TIMESERIES_COLUMN',
        'TIMESERIES_DATE_DUPLICATE',
        'TIMESERIES_DATE_COVERAGE',
      ]),
    );
  });
});
