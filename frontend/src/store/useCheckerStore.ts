import { create } from 'zustand';
import type { Diagnostic } from '../types/checker';

interface CheckerState {
  diagnostics: Diagnostic[];
  drawerOpen: boolean;
  activeTab: 'error' | 'warning' | 'missing' | 'logs';
  setDiagnostics: (diagnostics: Diagnostic[]) => void;
  setDrawerOpen: (open: boolean) => void;
  setActiveTab: (tab: CheckerState['activeTab']) => void;
}

export const useCheckerStore = create<CheckerState>((set) => ({
  diagnostics: [],
  drawerOpen: false,
  activeTab: 'error',
  setDiagnostics: (diagnostics) => set({ diagnostics }),
  setDrawerOpen: (drawerOpen) => set({ drawerOpen }),
  setActiveTab: (activeTab) => set({ activeTab }),
}));
