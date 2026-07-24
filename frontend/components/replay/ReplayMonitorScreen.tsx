'use client';

import { useEffect, useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { getReplay } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type { DriverSummary, ReplayLap, ReplayResponse } from '@/lib/api/types';
import { deviationMax } from '@/lib/replay/replay';
import { usePlayback } from '@/lib/replay/usePlayback';
import { useReplay } from '@/lib/state/replay';

import { AlertBanner } from './AlertBanner';
import { DeviationGauge } from './DeviationGauge';
import { DeviationTimeline } from './DeviationTimeline';
import { PerLapRadarPanel } from './PerLapRadarPanel';
import { PlaybackControls } from './PlaybackControls';

// Screen 6 root (master §10.7). SELECTION: the session is a UUID input — no session-catalog endpoint
// exists (div #77); the driver comes from /drivers (SSR'd by the page) or a UUID input if that failed.
// The replay is fetched CLIENT-side on demand (the session id is user input, not an SSR param), so a
// fetch failure surfaces via ErrorCard + a working onRetry=load — the client-triggered equivalent of
// the SSR PageError wrapper (same red-border/monospace pattern). A fresh checkout 503s honestly.
type FetchState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'result'; res: ApiResult<ReplayResponse> };

export function ReplayMonitorScreen({ initialDrivers }: { initialDrivers: DriverSummary[] }) {
  const [sessionId, setSessionId] = useState('');
  const [driverId, setDriverId] = useState(initialDrivers[0]?.id ?? '');
  const [state, setState] = useState<FetchState>({ kind: 'idle' });
  const reset = useReplay((s) => s.reset);

  const canLoad = sessionId.trim().length > 0 && driverId.trim().length > 0;

  async function load() {
    if (!canLoad) return;
    setState({ kind: 'loading' });
    reset(); // rewind playback for the new replay
    const res = await getReplay(sessionId.trim(), driverId.trim());
    setState({ kind: 'result', res });
  }

  return (
    <main className="mx-auto max-w-5xl p-6" data-testid="replay-screen">
      <h1 className="mb-4 font-mono text-lg uppercase tracking-wide text-accent">Race Replay Monitor</h1>

      {/* Selection: no session catalog endpoint → UUID input (div #77); driver from /drivers. */}
      <div className="mb-6 flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] uppercase text-text-secondary">Session id</span>
          <input
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            placeholder="Enter a session id (UUID)"
            className="w-72 rounded border border-border bg-card px-2 py-1 font-mono text-xs text-text-primary"
            data-testid="session-input"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] uppercase text-text-secondary">Driver</span>
          {initialDrivers.length > 0 ? (
            <select
              value={driverId}
              onChange={(e) => setDriverId(e.target.value)}
              className="w-40 rounded border border-border bg-card px-2 py-1 font-mono text-xs text-text-primary"
              data-testid="driver-select"
            >
              {initialDrivers.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.code}
                </option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={driverId}
              onChange={(e) => setDriverId(e.target.value)}
              placeholder="Driver id (UUID)"
              className="w-40 rounded border border-border bg-card px-2 py-1 font-mono text-xs text-text-primary"
              data-testid="driver-input"
            />
          )}
        </label>
        <button
          type="button"
          onClick={load}
          disabled={!canLoad}
          className="rounded border border-border px-3 py-1 font-mono text-xs text-accent transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-40"
          data-testid="load-replay"
        >
          Load replay
        </button>
        <p className="w-full font-mono text-[9px] text-text-secondary" data-testid="no-catalog-note">
          No session catalog endpoint — enter a session id.
        </p>
      </div>

      {state.kind === 'loading' ? <Skeleton className="h-24" label="Loading replay" /> : null}
      {state.kind === 'result' ? <ReplayResult res={state.res} onRetry={load} /> : null}
    </main>
  );
}

function ReplayResult({ res, onRetry }: { res: ApiResult<ReplayResponse>; onRetry: () => void }) {
  if (!res.ok) {
    // 404 / NOT_FOUND is honest absence → EmptyState, not an error. Everything else (incl. 503
    // HEALTH_FAIL / DB_UNAVAILABLE, UI_NETWORK/TIMEOUT/SCHEMA_MISMATCH) → ErrorCard; Retry shows for
    // the retryable codes (ErrorCard.isRetryable).
    if (res.status === 404 || res.code === 'NOT_FOUND') {
      return <EmptyState message="No replay for this session + driver." />;
    }
    return <ErrorCard code={res.code} detail={res.detail} onRetry={onRetry} />;
  }
  const laps = res.data.laps ?? [];
  if (laps.length === 0) {
    return <EmptyState message="No replay for this session + driver." />;
  }
  return <ReplayMonitor laps={laps} partial={res.partial} />;
}

function ReplayMonitor({ laps, partial }: { laps: ReplayLap[]; partial: boolean }) {
  const lapIndex = useReplay((s) => s.lapIndex);
  const setLapIndex = useReplay((s) => s.setLapIndex);
  usePlayback(laps.length);

  // Clamp if a shorter replay is loaded after a longer one (keeps the index in range).
  useEffect(() => {
    if (lapIndex > laps.length - 1) setLapIndex(laps.length - 1);
  }, [lapIndex, laps.length, setLapIndex]);

  const idx = Math.min(lapIndex, laps.length - 1);
  const current = laps[idx] ?? null;
  const max = deviationMax(laps);

  return (
    <div className="space-y-4" data-testid="replay-monitor">
      {partial ? (
        <div
          data-testid="partial-notice"
          className="rounded border border-l-4 border-border border-l-confidence-medium bg-card p-2 font-mono text-[10px] text-confidence-medium"
        >
          Showing a partial replay (some laps were truncated).
        </div>
      ) : null}

      <AlertBanner laps={laps} reachedIndex={idx} />

      <PlaybackControls lapCount={laps.length} currentLapNumber={current?.lap_number ?? null} />

      <div className="grid gap-4 md:grid-cols-[1fr_1.4fr]">
        <DeviationGauge
          score={current?.deviation_score ?? null}
          state={current?.alert_state ?? 'INACTIVE'}
          max={max}
        />
        <DeviationTimeline laps={laps} reachedIndex={idx} />
      </div>

      <PerLapRadarPanel lap={current} />
    </div>
  );
}
