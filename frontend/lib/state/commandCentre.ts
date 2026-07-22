import { create } from 'zustand';

import { DEFAULT_CIRCUIT } from '@/lib/design/circuits';

// Global UI state for Screen 1 (master §9 — Zustand for driver selection / circuit filter).
// Degradation-sensitivity sorting is intentionally absent — no per-driver scalar is exposed
// (F16 spec div #50); the control is rendered disabled, not silently missing.
export type SortKey = 'compatibility' | 'stability';

type CommandCentreState = {
  circuit: string;
  sortKey: SortKey;
  setCircuit: (circuit: string) => void;
  setSortKey: (sortKey: SortKey) => void;
};

export const useCommandCentre = create<CommandCentreState>((set) => ({
  circuit: DEFAULT_CIRCUIT,
  sortKey: 'compatibility',
  setCircuit: (circuit) => set({ circuit }),
  setSortKey: (sortKey) => set({ sortKey }),
}));
