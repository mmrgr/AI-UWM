import { create } from 'zustand';
import type { SimulationResponse } from '../types/results';

export interface ScenarioSnapshot {
  id: string;
  name: string;
  createdAt: string;
  result: SimulationResponse;
}

interface ScenarioState {
  scenarios: ScenarioSnapshot[];
  addScenario: (name: string, result: SimulationResponse) => void;
  removeScenario: (id: string) => void;
}

export const useScenarioStore = create<ScenarioState>((set) => ({
  scenarios: [],
  addScenario: (name, result) =>
    set((state) => ({
      scenarios: [
        ...state.scenarios,
        {
          id: crypto.randomUUID(),
          name,
          createdAt: new Date().toISOString(),
          result,
        },
      ],
    })),
  removeScenario: (id) =>
    set((state) => ({ scenarios: state.scenarios.filter((item) => item.id !== id) })),
}));
