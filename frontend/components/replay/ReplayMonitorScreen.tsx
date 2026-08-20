'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { getReplay, getSessions } from '@/lib/api/endpoints';
import type { AnyErrorCode, ApiResult } from '@/lib/api/errors';
import type { DriverSummary, ReplayLap, ReplayResponse, SessionSummary } from '@/lib/api/types';
import { teamColour } from '@/lib/design/teams';
import { gaugeMax, stintThreshold } from '@/lib/replay/replay';
import { usePlayback } from '@/lib/replay/usePlayback';
import { useReplay } from '@/lib/state/replay';

import { AlertBanner } from './AlertBanner';
import { DeviationGauge } from './DeviationGauge';
import { DeviationTimeline } from './DeviationTimeline';
import { DriverReplayGrid, type GridDriver } from './DriverReplayGrid';
import { FingerprintDriftPanel } from './FingerprintDriftPanel';
import { PlaybackControls } from './PlaybackControls';

// Screen 6 root (master §10.7). SELECTION now runs off the /sessions catalog (div #77 closed):
// session + driver are chip pickers over real persisted data — the raw-UUID input survives ONLY as
// the honest fallback when the catalog itself is unavailable (503/network), where typing an id can
// still work. MODE (div #76): "Single driver" is the F19a monitor; "All drivers" fans getReplay out
// over the session's replayable drivers (F19b). Replays are fetched client-side; every fetch keeps
// the four-state ApiResult discipline, and a fresh checkout (empty DB) shows an honest empty
// catalog, not an outage.
type Mode = 'single' | 'grid';
type CatalogState =
  | { kind: 'loading' }
  | { kind: 'error'; code: AnyErrorCode; detail?: string }
  | { kind: 'ready'; sessions: SessionSummary[] };
type FetchState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'result'; res: ApiResult<ReplayResponse> };

export function ReplayMonitorScreen({ initialDrivers }: { initialDrivers: DriverSummary[] }) {
  const [catalog, setCatalog] = useState<CatalogState>({ kind: 'loading' });
  const [mode, setMode] = useState<Mode>('single');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [driverCode, setDriverCode] = useState<string | null>(null);
  const [single, setSingle] = useState<FetchState>({ kind: 'idle' });
  // Manual fallback (catalog unavailable only): typed session id + driver.
  const [manualSession, setManualSession] = useState('');
  const [manualDriver, setManualDriver] = useState(initialDrivers[0]?.id ?? '');
  const [manualGridSession, setManualGridSession] = useState<string | null>(null);
  const reset = useReplay((s) => s.reset);

  const identities = useMemo(
    () => new Map(initialDrivers.map((d) => [d.code, d])),
    [initialDrivers],
  );

  const loadCatalog = useCallback(() => {
    setCatalog({ kind: 'loading' });
    void getSessions().then((r) => {
      if (r.ok) {
        setCatalog({ kind: 'ready', sessions: r.data });
        // Auto-select the newest session + its first driver so the monitor is one click away.
        const first = r.data[0] ?? null;
        setSessionId(first?.id ?? null);
        setDriverCode(first?.drivers?.[0]?.code ?? null);
      } else {
        setCatalog({ kind: 'error', code: r.code, detail: r.detail });
      }
    });
  }, []);

  useEffect(() => {
    loadCatalog();
  }, [loadCatalog]);

  const session = useMemo(
    () =>
      catalog.kind === 'ready'
        ? (catalog.sessions.find((s) => s.id === sessionId) ?? null)
        : null,
    [catalog, sessionId],
  );

  const loadSingle = useCallback(
    async (sid: string, driver: string) => {
      reset(); // rewind playback for the new replay
      setSingle({ kind: 'loading' });
      const res = await getReplay(sid, driver);
      setSingle({ kind: 'result', res });
    },
    [reset],
  );

  // Catalog-driven auto-fetch: a (session, driver) chip selection IS the load action.
  useEffect(() => {
    if (mode !== 'single' || catalog.kind !== 'ready') return;
    if (sessionId == null || driverCode == null) return;
    void loadSingle(sessionId, driverCode);
  }, [mode, catalog.kind, sessionId, driverCode, loadSingle]);

  function selectSession(s: SessionSummary) {
    setSessionId(s.id);
    // Keep the driver if they also ran this session; else fall to its first driver.
    const codes = (s.drivers ?? []).map((d) => d.code);
    setDriverCode(driverCode != null && codes.includes(driverCode) ? driverCode : (codes[0] ?? null));
    reset();
  }

  function switchMode(next: Mode) {
    if (next === mode) return;
    setMode(next);
    setSingle({ kind: 'idle' });
    setManualGridSession(null);
    reset();
  }

  // Manual fallback load (catalog unavailable): the pre-catalog flow, kept honest.
  const manualCanLoad =
    manualSession.trim().length > 0 &&
    (mode === 'grid' ? initialDrivers.length > 0 : manualDriver.trim().length > 0);

  function manualLoad() {
    if (!manualCanLoad) return;
    reset();
    if (mode === 'grid') {
      setManualGridSession(manualSession.trim());
      return;
    }
    void loadSingle(manualSession.trim(), manualDriver.trim());
  }

  // Grid drivers: the selected session's replayable codes (team colour joined from /drivers).
  const gridDrivers: GridDriver[] = useMemo(() => {
    if (catalog.kind === 'ready' && session != null) {
      return (session.drivers ?? []).map((d) => ({
        code: d.code,
        team: identities.get(d.code)?.team ?? null,
      }));
    }
    return initialDrivers.map((d) => ({ code: d.code, team: d.team ?? null }));
  }, [catalog.kind, session, identities, initialDrivers]);

  const driverColour = teamColour(
    driverCode != null ? identities.get(driverCode)?.team : null,
  );

  return (
    <div data-testid="replay-screen">
      <header className="mb-5">
        <h1 className="font-display text-[34px] font-bold uppercase leading-none tracking-[0.03em] text-text-primary">
          Race Replay Monitor
        </h1>
        <p className="mt-2 text-[13px] text-text-secondary">
          A persisted session replayed lap by lap through the behavioural deviation monitor.
          Deviation, threshold and alerts appear exactly as they would have fired live.
        </p>
      </header>

      {/* Selection — sessions from the /sessions catalog (div #77 closed), drivers per session. */}
      <div className="mb-4 rounded-md border border-border bg-card p-4">
        {catalog.kind === 'loading' ? (
          <Skeleton className="h-16" label="Loading session catalog" />
        ) : catalog.kind === 'error' ? (
          <div className="space-y-3" data-testid="catalog-fallback">
            <ErrorCard code={catalog.code} detail={catalog.detail} onRetry={loadCatalog} />
            <p className="font-mono text-[9px] text-text-secondary" data-testid="catalog-unavailable-note">
              Session catalog unavailable. Enter a session id manually.
            </p>
            <ModeToggle mode={mode} onSwitch={switchMode} />
            <div className="flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1">
                <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                  Session id
                </span>
                <input
                  type="text"
                  value={manualSession}
                  onChange={(e) => setManualSession(e.target.value)}
                  placeholder="Enter a session id (UUID)"
                  className="w-72 rounded border border-border bg-card px-2 py-1 font-mono text-xs text-text-primary"
                  data-testid="session-input"
                />
              </label>
              {mode === 'single' ? (
                <label className="flex flex-col gap-1">
                  <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                    Driver
                  </span>
                  {initialDrivers.length > 0 ? (
                    <select
                      value={manualDriver}
                      onChange={(e) => setManualDriver(e.target.value)}
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
                      value={manualDriver}
                      onChange={(e) => setManualDriver(e.target.value)}
                      placeholder="Driver id (UUID)"
                      className="w-40 rounded border border-border bg-card px-2 py-1 font-mono text-xs text-text-primary"
                      data-testid="driver-input"
                    />
                  )}
                </label>
              ) : null}
              <button
                type="button"
                onClick={manualLoad}
                disabled={!manualCanLoad}
                className="rounded border border-border px-3 py-1 font-mono text-xs text-accent transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-40"
                data-testid="load-replay"
              >
                Load replay
              </button>
              {mode === 'grid' && initialDrivers.length === 0 ? (
                <p className="w-full font-mono text-[9px] text-alert-ink" data-testid="grid-no-drivers-note">
                  Driver list unavailable. Grid mode needs /drivers.
                </p>
              ) : null}
            </div>
          </div>
        ) : catalog.sessions.length === 0 ? (
          <p className="font-mono text-[11px] text-text-secondary" data-testid="catalog-empty">
            No sessions persisted yet. Seed the demo race or run an ingestion batch, then reload.
          </p>
        ) : (
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                Session
              </p>
              <div className="flex flex-wrap gap-1.5" data-testid="session-chips">
                {catalog.sessions.map((s) => {
                  const selected = s.id === sessionId;
                  return (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => selectSession(s)}
                      data-testid={`session-chip-${s.id}`}
                      data-selected={selected || undefined}
                      title={`Round ${s.round} · ${(s.drivers ?? []).length} drivers`}
                      className={`rounded border px-2.5 py-1 font-mono text-[11px] transition-colors focus:outline-none focus:ring-1 focus:ring-accent ${
                        selected
                          ? 'border-accent-deep bg-accent-fill text-text-primary'
                          : 'border-border text-text-secondary hover:border-text-secondary hover:text-text-primary'
                      }`}
                    >
                      {s.circuit} {s.season} · {s.session_type}
                    </button>
                  );
                })}
              </div>
            </div>
            {mode === 'single' && session != null ? (
              <div>
                <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                  Driver
                </p>
                <div className="flex flex-wrap gap-1.5" data-testid="driver-chips">
                  {(session.drivers ?? []).map((d) => {
                    const selected = d.code === driverCode;
                    return (
                      <button
                        key={d.code}
                        type="button"
                        onClick={() => {
                          setDriverCode(d.code);
                          reset();
                        }}
                        data-testid={`driver-chip-${d.code}`}
                        data-selected={selected || undefined}
                        title={`${d.n_laps} replayable laps`}
                        className={`rounded border px-2.5 py-1 font-mono text-[11px] transition-colors focus:outline-none focus:ring-1 focus:ring-accent ${
                          selected
                            ? 'border-accent-deep bg-accent-fill text-text-primary'
                            : 'border-border text-text-secondary hover:border-text-secondary hover:text-text-primary'
                        }`}
                      >
                        {d.code}
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : null}
            <ModeToggle mode={mode} onSwitch={switchMode} />
          </div>
        )}
      </div>

      {mode === 'single' ? (
        <>
          {single.kind === 'loading' ? <Skeleton className="h-24" label="Loading replay" /> : null}
          {single.kind === 'result' ? (
            <ReplayResult
              res={single.res}
              colour={driverColour}
              onRetry={() => {
                if (catalog.kind === 'ready' && sessionId != null && driverCode != null) {
                  void loadSingle(sessionId, driverCode);
                } else {
                  manualLoad();
                }
              }}
            />
          ) : null}
        </>
      ) : catalog.kind === 'ready' && session != null ? (
        <DriverReplayGrid key={session.id} sessionId={session.id} drivers={gridDrivers} />
      ) : manualGridSession != null ? (
        <DriverReplayGrid key={manualGridSession} sessionId={manualGridSession} drivers={gridDrivers} />
      ) : null}
    </div>
  );
}

function ModeToggle({ mode, onSwitch }: { mode: Mode; onSwitch: (m: Mode) => void }) {
  return (
    <div>
      <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
        Mode
      </p>
      <div className="flex gap-1.5" role="group" aria-label="Replay mode">
        {(
          [
            ['single', 'Single driver', 'mode-single'],
            ['grid', 'All drivers', 'mode-grid'],
          ] as const
        ).map(([m, label, testid]) => {
          const selected = mode === m;
          return (
            <button
              key={m}
              type="button"
              onClick={() => onSwitch(m)}
              data-testid={testid}
              data-selected={selected || undefined}
              className={`rounded border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.1em] transition-colors focus:outline-none focus:ring-1 focus:ring-accent ${
                selected
                  ? 'border-accent-deep bg-accent-fill text-text-primary'
                  : 'border-border text-text-secondary hover:border-text-secondary hover:text-text-primary'
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ReplayResult({
  res,
  colour,
  onRetry,
}: {
  res: ApiResult<ReplayResponse>;
  colour: string;
  onRetry: () => void;
}) {
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
  return <ReplayMonitor laps={laps} partial={res.partial} colour={colour} />;
}

function ReplayMonitor({
  laps,
  partial,
  colour,
}: {
  laps: ReplayLap[];
  partial: boolean;
  colour: string;
}) {
  const lapIndex = useReplay((s) => s.lapIndex);
  const setLapIndex = useReplay((s) => s.setLapIndex);
  usePlayback(laps.length);

  // Clamp if a shorter replay is loaded after a longer one (keeps the index in range).
  useEffect(() => {
    if (lapIndex > laps.length - 1) setLapIndex(laps.length - 1);
  }, [lapIndex, laps.length, setLapIndex]);

  const idx = Math.min(lapIndex, laps.length - 1);
  const current = laps[idx] ?? null;
  const max = gaugeMax(laps);
  const tau = stintThreshold(laps);

  return (
    <div className="space-y-4" data-testid="replay-monitor">
      {partial ? (
        <div
          data-testid="partial-notice"
          className="rounded border border-l-4 border-border border-l-alert-ink bg-alert-fill p-2 font-mono text-[10px] text-alert-ink"
        >
          Showing a partial replay (some laps were truncated).
        </div>
      ) : null}

      <AlertBanner lap={current} />

      <section className="rounded-md border border-border bg-card p-4">
        <PlaybackControls lapCount={laps.length} currentLapNumber={current?.lap_number ?? null} />
      </section>

      <div className="grid gap-4 md:grid-cols-[minmax(220px,1fr)_2fr]">
        <section className="rounded-md border border-border bg-card p-4">
          <DeviationGauge
            score={current?.deviation_score ?? null}
            state={current?.alert_state ?? 'INACTIVE'}
            max={max}
            threshold={tau}
          />
        </section>
        <section className="rounded-md border border-border bg-card p-4">
          <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
            Deviation timeline
          </p>
          <DeviationTimeline laps={laps} reachedIndex={idx} />
        </section>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <section className="rounded-md border border-border bg-card p-4">
          <FingerprintDriftPanel laps={laps} reachedIndex={idx} colour={colour} />
        </section>
        <section className="rounded-md border border-border bg-card p-4">
          <MonitorStatus laps={laps} current={current} tau={tau} />
        </section>
      </div>
    </div>
  );
}

// A compact live read of the monitor as of the playback position — every number is served data.
function MonitorStatus({
  laps,
  current,
  tau,
}: {
  laps: readonly ReplayLap[];
  current: ReplayLap | null;
  tau: number | null;
}) {
  const scored = laps.filter((l) => typeof l.deviation_score === 'number').length;
  const alerts = laps.filter((l) => l.alert_state === 'ALERT').length;
  const rows: Array<[string, string]> = [
    ['Monitor state', current?.alert_state ?? '—'],
    [
      'Deviation score',
      typeof current?.deviation_score === 'number' ? current.deviation_score.toFixed(3) : '—',
    ],
    ['Alert threshold τ', tau != null ? tau.toFixed(3) : '—'],
    ['Scored laps', `${scored} of ${laps.length}`],
    ['Laps in ALERT', String(alerts)],
  ];
  return (
    <div data-testid="monitor-status">
      <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
        Monitor status
      </p>
      <dl className="space-y-1.5">
        {rows.map(([label, value]) => (
          <div key={label} className="flex items-baseline justify-between gap-4">
            <dt className="font-mono text-[10px] uppercase tracking-[0.1em] text-text-secondary">
              {label}
            </dt>
            <dd className="font-mono text-[12px] tabular-nums text-text-primary">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 font-mono text-[9px] leading-relaxed text-text-secondary">
        Scores are causal. Lap k uses only laps 1 to k, exactly what a live monitor would have known.
      </p>
    </div>
  );
}
