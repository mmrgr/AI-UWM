import { create } from 'zustand';

interface FocusRequest {
  nodeId: string;
  field?: string;
  nonce: number;
}

interface CanvasState {
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  zoom: number;
  focusRequest: FocusRequest | null;
  selectNode: (id: string | null) => void;
  selectEdge: (id: string | null) => void;
  setZoom: (zoom: number) => void;
  focusNode: (nodeId: string, field?: string) => void;
}

export const useCanvasStore = create<CanvasState>((set) => ({
  selectedNodeId: null,
  selectedEdgeId: null,
  zoom: 1,
  focusRequest: null,
  selectNode: (selectedNodeId) =>
    set({ selectedNodeId, selectedEdgeId: selectedNodeId ? null : undefined }),
  selectEdge: (selectedEdgeId) =>
    set({ selectedEdgeId, selectedNodeId: selectedEdgeId ? null : undefined }),
  setZoom: (zoom) => set({ zoom }),
  focusNode: (nodeId, field) =>
    set({
      selectedNodeId: nodeId,
      selectedEdgeId: null,
      focusRequest: { nodeId, field, nonce: Date.now() },
    }),
}));
