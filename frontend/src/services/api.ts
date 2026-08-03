import type {
  CalibrationResponse,
  DecisionSupportResponse,
  OptimizationResponse,
  SimulationResponse,
  UncertaintyResponse,
} from '../types/results';
import type { TimeSeriesRow, WaterMetProject } from '../types/project';

interface ProjectPayload {
  project: WaterMetProject;
  timeseries: TimeSeriesRow[];
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const body = (await response.json()) as unknown;
  if (!response.ok) {
    const detail =
      typeof body === 'object' &&
      body !== null &&
      'detail' in body &&
      typeof body.detail === 'string'
        ? body.detail
        : `请求失败：HTTP ${response.status}`;
    throw new Error(detail);
  }
  return body as T;
}

export async function loadDemoProject(): Promise<ProjectPayload> {
  return requestJson<ProjectPayload>('/api/project/demo');
}

export async function validateProject(payload: ProjectPayload): Promise<{
  valid: boolean;
  components: number;
  local_areas: number;
  days: number;
}> {
  return requestJson('/api/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runSimulation(payload: ProjectPayload): Promise<SimulationResponse> {
  return requestJson<SimulationResponse>('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runDecisionSupport(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<DecisionSupportResponse> {
  return requestJson<DecisionSupportResponse>('/api/dss', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runCalibration(
  payload: ProjectPayload & {
    specification: Record<string, unknown>;
    observed: Record<string, unknown>[];
  },
): Promise<CalibrationResponse> {
  return requestJson<CalibrationResponse>('/api/calibrate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runUncertainty(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<UncertaintyResponse> {
  return requestJson<UncertaintyResponse>('/api/uncertainty', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runOptimization(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<OptimizationResponse> {
  return requestJson<OptimizationResponse>('/api/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function exportSimulation(payload: ProjectPayload): Promise<Blob> {
  const response = await fetch('/api/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = (await response.json()) as { detail?: string };
    throw new Error(body.detail ?? `导出失败：HTTP ${response.status}`);
  }
  return response.blob();
}
