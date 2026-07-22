'use client';

import { useEffect, useMemo, useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { getSetup, getStability } from '@/lib/api/endpoints';
import type { DriverSummary, SetupResponse } from '@/lib/api/types';
import { useCommandCentre } from '@/lib/state/commandCentre';

import { DriverCard, type Metric } from './DriverCard';
import { CircuitSelector } from './CircuitSelector';
import { SortControls } from './SortControls';

const LOADING: Metric = { value: null, loading: true, error: false };

// Circuit-compatibility proxy (F16 spec div #47): the mean per-parameter reliability of the
// driver's setup recommendation for the selected circuit. A 404 (no data) → null → "—" on the card.
function meanConfidence(setup: SetupResponse): number | null {
  const ps = setup.parameters;
  if (!ps.length) return null;
  return ps.reduce((sum, p) => sum + (p.confidence ?? 0), 0) / ps.length;
}

function initialMap(drivers: DriverSummary[]): Record<string, Metric> {
  return Object.fromEntries(drivers.map((d) => [d.code, LOADING]));
}

export function DriverGrid({
  initialDrivers,
  animate = true,
}: {
  initialDrivers: DriverSummary[];
  animate?: boolean;
}) {
  const sortKey = useCommandCentre((s) => s.sortKey);
  const circuit = useCommandCentre((s) => s.circuit);
  const [stability, setStability] = useState<Record<string, Metric>>(() => initialMap(initialDrivers));
  const [compat, setCompat] = useState<Record<string, Metric>>(() => initialMap(initialDrivers));

  // Stability is circuit-independent → fetched once per driver set.
  useEffect(() => {
    let alive = true;
    setStability(initialMap(initialDrivers));
    for (const d of initialDrivers) {
      void getStability(d.code).then((r) => {
        if (!alive) return;
        setStability((prev) => ({
          ...prev,
          [d.code]: r.ok
            ? { value: r.data.stability_score ?? null, loading: false, error: false }
            : { value: null, loading: false, error: true },
        }));
      });
    }
    return () => {
      alive = false;
    };
  }, [initialDrivers]);

  // Compatibility proxy is per-circuit → refetched when the circuit changes.
  useEffect(() => {
    let alive = true;
    setCompat(initialMap(initialDrivers));
    for (const d of initialDrivers) {
      void getSetup(d.code, circuit).then((r) => {
        if (!alive) return;
        // A 404 is honest "no data" (→ "—"), NOT an error state.
        const isMissing = !r.ok && r.code === 'NOT_FOUND';
        setCompat((prev) => ({
          ...prev,
          [d.code]: {
            value: r.ok ? meanConfidence(r.data) : null,
            loading: false,
            error: !r.ok && !isMissing,
          },
        }));
      });
    }
    return () => {
      alive = false;
    };
  }, [initialDrivers, circuit]);

  const sorted = useMemo(() => {
    const scoreOf = (code: string): number => {
      const m = sortKey === 'compatibility' ? compat[code] : stability[code];
      return m?.value ?? -1; // unknown sinks to the bottom, deterministically
    };
    return [...initialDrivers].sort((a, b) => scoreOf(b.code) - scoreOf(a.code));
  }, [initialDrivers, sortKey, compat, stability]);

  if (initialDrivers.length === 0) {
    return <EmptyState message="No driver data yet." />;
  }

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <CircuitSelector />
        <SortControls />
      </div>
      <div
        className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4"
        data-testid="driver-grid"
      >
        {sorted.map((d) => (
          <DriverCard
            key={d.code}
            driver={d}
            animate={animate}
            compatibility={compat[d.code] ?? LOADING}
            stability={stability[d.code] ?? LOADING}
          />
        ))}
      </div>
    </div>
  );
}
