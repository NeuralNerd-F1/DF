'use client';

import { CIRCUITS } from '@/lib/design/circuits';
import { useCommandCentre } from '@/lib/state/commandCentre';

// Circuit selector (master §10.2). Changing it re-transitions every card's radar (via the store).
export function CircuitSelector() {
  const circuit = useCommandCentre((s) => s.circuit);
  const setCircuit = useCommandCentre((s) => s.setCircuit);
  return (
    <label className="flex items-center gap-2 text-xs text-text-secondary">
      Circuit
      <select
        aria-label="Circuit"
        value={circuit}
        onChange={(e) => setCircuit(e.target.value)}
        className="rounded border border-border bg-card px-2 py-1 font-mono text-xs text-text-primary"
      >
        {CIRCUITS.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    </label>
  );
}
