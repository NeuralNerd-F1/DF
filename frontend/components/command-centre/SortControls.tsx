'use client';

import { Button } from '@/components/ds/Button';
import { useCommandCentre } from '@/lib/state/commandCentre';

// Sort controls (master §10.2). Compatibility + stability sort on live per-driver scalars.
// Degradation-sensitivity is present-but-disabled with a tooltip: no per-driver scalar exists
// (F16 spec div #50) — the option is surfaced honestly, never a fabricated ordering.
export function SortControls() {
  const sortKey = useCommandCentre((s) => s.sortKey);
  const setSortKey = useCommandCentre((s) => s.setSortKey);
  return (
    <div className="flex items-center gap-2 text-xs text-text-secondary" role="group" aria-label="Sort drivers">
      Sort
      <Button active={sortKey === 'compatibility'} onClick={() => setSortKey('compatibility')}>
        Compatibility
      </Button>
      <Button active={sortKey === 'stability'} onClick={() => setSortKey('stability')}>
        Stability
      </Button>
      <Button
        disabled
        aria-disabled
        title="Degradation-sensitivity sort needs a per-driver metric the API does not expose yet."
      >
        Degradation
      </Button>
    </div>
  );
}
