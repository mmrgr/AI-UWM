import { create } from 'zustand';
import type { ResultFrequency, SimulationResponse } from '../types/results';

interface ResultState {
  status: 'idle' | 'validating' | 'running' | 'success' | 'error';
  result: SimulationResponse | null;
  logs: string[];
  error: string | null;
  frequency: ResultFrequency;
  setStatus: (status: ResultState['status']) => void;
  setResult: (result: SimulationResponse) => void;
  addLog: (message: string) => void;
  setError: (message: string) => void;
  setFrequency: (frequency: ResultFrequency) => void;
  clear: () => void;
}

export const useResultStore = create<ResultState>((set) => ({
  status: 'idle',
  result: null,
  logs: ['WaterMet² Studio 已就绪。'],
  error: null,
  frequency: 'daily',
  setStatus: (status) => set({ status }),
  setResult: (result) =>
    set((state) => ({
      status: 'success',
      result,
      error: null,
      logs: [...state.logs, '仿真完成，结果已载入。'],
    })),
  addLog: (message) => set((state) => ({ logs: [...state.logs, message] })),
  setError: (error) =>
    set((state) => ({
      status: 'error',
      error,
      logs: [...state.logs, `错误：${error}`],
    })),
  setFrequency: (frequency) => set({ frequency }),
  clear: () => set({ status: 'idle', result: null, error: null }),
}));
