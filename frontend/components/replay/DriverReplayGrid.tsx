'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { getReplay } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type { DriverSummary, ReplayResponse } from '@/lib/api/types';
import { deviationMax, firstAlertLap, lapsInAlert } from '@/lib/replay/replay';
import { usePlayback } from '@/lib/replay/usePlayback';
import { useReplay } from '@/lib/state/replay';

import { DeviationGauge } from './DeviationGauge';
import { DeviationTimeline } from './DeviationTimeline';
import { PlaybackControls } from './PlaybackControls';

// Screen 6 · 8-driver mode (master §10.7, div #76 — the F19b contingency). There is NO batch replay
// endpoint, so this is an N-parallel getReplay FAN-OUT: one request per /drivers driver, each cell an
// INDEPENDENT four-state panel (loading / empty / error / success) that resolves and retries on its
// own. All cells share ONE playback position (the global store) so Play/Pause/Speed/Scrub monitors
// every driver in lock-step; the shared scrubber spans the longest replay and each cell clamps to its
// own lap count.
type CellState = { kind: 'loading' } | { kind: 'result'; res: ApiResult<ReplayResponse> };

export function DriverReplayGrid({
  drivers,
  sessionId,
}: {
  drivers: DriverSummary[];
  sessionId: string;
}) {
  const [cells, setCells] = useState<CellState[]>(() => drivers.map(() => ({ kind: 'loading' })));
  const reset = useReplay((s) => s.reset);

  // Fetch (or re-fetch) a single cell — this is the per-cell Retry, keeping each panel independent.
  const fetchCell = useCallback(
    (i: number) => {
      const d = drivers[i];
      if (!d) return;
      setCells((prev) => {
        const next = prev.slice();
        next[i] = { kind: 'loading' };
        return next;
      });
      getReplay(sessionId, d.id).then((res) => {
        setCells((prev) => {
          const next = prev.slice();
          next[i] = { kind: 'result', res };
          return next;
        });
      });
    },
    [drivers, sessionId],
  );

  // On mount / session change: rewind the shared playback and fan out one request per driver.
  useEffect(() => {
    reset();
    drivers.forEach((_, i) => fetchCell(i));
  }, [drivers, sessionId, reset, fetchCell]);

  // The shared scrubber spans the longest successful replay; each cell clamps to its own length.
  const maxLaps = useMemo(
    () =>
      cells.reduce(
        (m, c) => (c.kind === 'result' && c.res.ok ? Math.max(m, (c.res.data.laps ?? []).length) : m),
        0,
      ),
    [cells],
  );
  usePlayback(maxLaps);

  return (
    <div className="space-y-4" data-testid="driver-grid">
      <PlaybackControls lapCount={maxLaps} currentLapNumber={null} />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {drivers.map((d, i) => (
          <DriverReplayCell key={d.id} driver={d} state={cells[i] ?? { kind: 'loading' }} onRetry={() => fetchCell(i)} />
        ))}
      </div>
    </div>
  );
}

function DriverReplayCell({
  driver,
  state,
  onRetry,
}: {
  driver: DriverSummary;
  state: CellState;
  onRetry: () => void;
}) {
  const lapIndex = useReplay((s) => s.lapIndex);

  return (
    <div
      data-testid="grid-cell"
      data-driver={driver.code}
      className="rounded-lg border border-border bg-card p-3"
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-xs uppercase text-text-primary">{driver.code}</span>
      </div>
      <CellBody state={state} lapIndex={lapIndex} onRetry={onRetry} />
    </div>
  );
}

function CellBody({
  state,
  lapIndex,
  onRetry,
}: {
  state: CellState;
  lapIndex: number;
  onRetry: () => void;
}) {
  if (state.kind === 'loading') return <Skeleton className="h-16" label="Loading" />;

  const res = state.res;
  if (!res.ok) {
    // Each cell surfaces its OWN honest state — 404/NOT_FOUND is absence (EmptyState), everything else
    // is an ErrorCard with a per-cell Retry (retryable codes only).
    if (res.status === 404 || res.code === 'NOT_FOUND') return <EmptyState message="No replay." />;
    return <ErrorCard code={res.code} detail={res.detail} onRetry={onRetry} />;
  }

  const laps = res.data.laps ?? [];
  if (laps.length === 0) return <EmptyState message="No replay." />;

  const idx = Math.min(lapIndex, laps.length - 1);
  const current = laps[idx] ?? null;
  const max = deviationMax(laps);
  const alertLap = firstAlertLap(laps, idx);

  return (
    <div className="space-y-2">
      {res.partial ? (
        <p className="font-mono text-[9px] text-confidence-medium" data-testid="cell-partial">
          partial replay
        </p>
      ) : null}
      {alertLap ? (
        <div
          role="alert"
          data-testid="cell-alert"
          className="rounded border-l-2 border-l-accent bg-card px-2 py-1 font-mono text-[9px] text-accent"
        >
          ALERT · lap {alertLap.lap_number} · {lapsInAlert(laps, idx)} in alert
        </div>
      ) : null}
      <DeviationGauge
        score={current?.deviation_score ?? null}
        state={current?.alert_state ?? 'INACTIVE'}
        max={max}
      />
      <DeviationTimeline laps={laps} reachedIndex={idx} />
    </div>
  );
}
