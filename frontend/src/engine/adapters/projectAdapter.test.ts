import { describe, expect, it } from 'vitest';
import { defaultProject } from '../../data/defaultProject';
import { graphToProject, projectToGraph } from './projectAdapter';

describe('projectAdapter', () => {
  it('round-trips the editable topology and component data', () => {
    const graph = projectToGraph(defaultProject);
    expect(graph.nodes.length).toBe(
      Object.keys(defaultProject.components).length +
      Object.keys(defaultProject.local_areas).length,
    );
    expect(graph.edges.length).toBe(7);

    const project = graphToProject(defaultProject, graph.nodes, graph.edges);
    expect(Object.keys(project.components)).toEqual(
      expect.arrayContaining(Object.keys(defaultProject.components)),
    );
    expect(project.supply_connections).toHaveLength(0);
    expect(project.supply_paths).toHaveLength(1);
    expect(project.wastewater_connections).toHaveLength(2);
  });
});
