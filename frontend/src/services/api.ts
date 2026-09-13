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

export async function runAiBaseline(payload: ProjectPayload): Promise<{
  comparison: Record<string, unknown>[];
}> {
  return requestJson('/api/ai/baseline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runAiSensitivity(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<{ method: string; metric: string; sensitivity: Record<string, unknown>[] }> {
  return requestJson('/api/ai/sensitivity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runAiProbabilisticThreshold(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<{ threshold_samples: Record<string, unknown>[] }> {
  return requestJson('/api/ai/probabilistic-threshold', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runAiExceedanceProbability(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<{ proposed_capacity_mw: number; exceedance_probability: number }> {
  return requestJson('/api/ai/exceedance-probability', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runAiBottlenecks(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<{
  constraint_boundaries: Record<string, unknown>[];
  interventions: Record<string, unknown>[];
}> {
  return requestJson('/api/ai/bottlenecks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runAiIntradayProxy(
  payload: ProjectPayload & { specification?: Record<string, unknown> },
): Promise<{
  profile: Record<string, unknown>[];
  hourly: Record<string, unknown>[];
}> {
  return requestJson('/api/ai/intraday-proxy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function runAiHourlyBoundary(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<{ scan: Record<string, unknown>[]; margins: Record<string, unknown>[] }> {
  return requestJson('/api/ai/hourly-boundary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export async function runAiStates(
  payload: ProjectPayload & { specification?: Record<string, unknown> },
): Promise<{ scenarios: Record<string, unknown>[] }> {
  return requestJson('/api/ai/states', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export async function runAiIndustrialControl(
  payload: ProjectPayload & { specification?: Record<string, unknown> },
): Promise<{ comparison: Record<string, unknown>[] }> {
  return requestJson('/api/ai/industrial-control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export async function runAiRobustness(
  payload: ProjectPayload & { specification: Record<string, unknown> },
): Promise<{ samples: Record<string, unknown>[] }> {
  return requestJson('/api/ai/robustness', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export async function runAiProvenance(payload: ProjectPayload): Promise<{ parameters: Record<string, unknown>[] }> {
  return requestJson('/api/ai/provenance', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
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
