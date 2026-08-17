'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { getReplay } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type { ReplayResponse } from '@/lib/api/types';
import { teamColour } from '@/lib/design/teams';
import { gaugeMax, stintThreshold } from '@/lib/replay/replay';
import { usePlayback } from '@/lib/replay/usePlayback';
import { useReplay } from '@/lib/state/replay';

import { DeviationGauge } from './DeviationGauge';
import { DeviationTimeline } from './DeviationTimeline';
import { PlaybackControls } from './PlaybackControls';

// Screen 6 · all-drivers mode (master §10.7, div #76 — F19b). There is NO batch replay endpoint,
// so this is an N-parallel getReplay FAN-OUT: one request per replayable driver in the selected
// session (their codes come from the /sessions catalog), each cell an INDEPENDENT four-state panel
// that resolves and retries on its own. All cells share ONE playback position (the global store) so
// Play/Pause/Speed/Scrub monitors every driver in lock-step; the shared scrubber spans the longest
// replay and each cell clamps to its own lap count.
export type GridDriver = { code: string; team?: string | null };

type CellState = { kind: 'loading' } | { kind: 'result'; res: ApiResult<ReplayResponse> };

export function DriverReplayGrid({
  drivers,
  sessionId,
}: {
  drivers: GridDriver[];
  sessionId: string;
}) {
  const [cells, setCells] = useState<CellState[]>(() => drivers.map(() => ({ kind: 'loading' })));
  const reset = useReplay((s) => s.reset);

  // Fetch (or re-fetch) a single cell — this is the per-cell Retry, keeping each panel independent.
  const fetchCell = useCallback(
    (i: number): Promise<void> => {
      const d = drivers[i];
      if (!d) return Promise.resolve();
      setCells((prev) => {
        const next = prev.slice();
        next[i] = { kind: 'loading' };
        return next;
      });
      return getReplay(sessionId, d.code).then((res) => {
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
  // The fan-out is a rolling queue, not N-wide parallel: each replay costs ~1s of CPU on the
  // single-worker API, so unbounded parallelism just stacks tail latency into client timeouts.
  // Two in flight keeps cells filling progressively while every request finishes promptly.
  useEffect(() => {
    reset();
    let cancelled = false;
    let cursor = 0;
    const next = (): Promise<void> => {
      if (cancelled || cursor >= drivers.length) return Promise.resolve();
      const i = cursor++;
      return fetchCell(i).then(next);
    };
    void next();
    void next();
    return () => {
      cancelled = true;
    };
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
      <section className="rounded-md border border-border bg-card p-4">
        <PlaybackControls lapCount={maxLaps} currentLapNumber={null} />
      </section>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {drivers.map((d, i) => (
          <DriverReplayCell
            key={d.code}
            driver={d}
            state={cells[i] ?? { kind: 'loading' }}
            onRetry={() => fetchCell(i)}
          />
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
  driver: GridDriver;
  state: CellState;
  onRetry: () => void;
}) {
  const lapIndex = useReplay((s) => s.lapIndex);

  const colour = teamColour(driver.team);
  const ruleId = `grid-rule-${driver.code}`;

  return (
    <div
      data-testid="grid-cell"
      data-driver={driver.code}
      className="overflow-hidden rounded-md border border-border bg-card"
    >
      {/* Team rule — solid at the left, fading right (SVG gradient, no inline style). */}
      <svg viewBox="0 0 100 3" preserveAspectRatio="none" className="h-[3px] w-full" aria-hidden="true">
        <defs>
          <linearGradient id={ruleId} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor={colour} />
            <stop offset="0.6" stopColor={colour} stopOpacity={0.9} />
            <stop offset="1" stopColor={colour} stopOpacity={0.15} />
          </linearGradient>
        </defs>
        <rect width={100} height={3} fill={`url(#${ruleId})`} />
      </svg>
      <div className="p-3">
        <div className="mb-2 flex items-baseline justify-between">
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-text-primary">
            {driver.code}
          </span>
          {driver.team ? (
            <span className="font-mono text-[8.5px] uppercase tracking-[0.14em] text-text-secondary">
              {driver.team}
            </span>
          ) : null}
        </div>
        <CellBody state={state} lapIndex={lapIndex} onRetry={onRetry} />
      </div>
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
  const max = gaugeMax(laps);
  const tau = stintThreshold(laps);
  const alert = current?.alert ?? null;

  return (
    <div className="space-y-2">
      {res.partial ? (
        <p className="font-mono text-[9px] text-alert-ink" data-testid="cell-partial">
          partial replay
        </p>
      ) : null}
      {alert ? (
        <div
          role="alert"
          data-testid="cell-alert"
          className="rounded border-l-2 border-l-alert-ink bg-alert-fill px-2 py-1 font-mono text-[9px] text-alert-ink"
        >
          ALERT · lap {alert.alert_lap} · {alert.alert_confidence} · {alert.laps_since_threshold_crossed}{' '}
          since crossing
        </div>
      ) : null}
      <DeviationGauge
        score={current?.deviation_score ?? null}
        state={current?.alert_state ?? 'INACTIVE'}
        max={max}
        threshold={tau}
      />
      <DeviationTimeline laps={laps} reachedIndex={idx} compact />
    </div>
  );
}
