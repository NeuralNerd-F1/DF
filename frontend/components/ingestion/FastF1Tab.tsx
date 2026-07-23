'use client';

import { useEffect, useState } from 'react';

import { Button } from '@/components/ds/Button';
import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { getDrivers } from '@/lib/api/endpoints';
import type { DriverSummary } from '@/lib/api/types';

import { PipelineProgress, type PipelinePhase } from './PipelineProgress';
import { useIngestionJob } from './useIngestionJob';

// FastF1 Session tab (master §10.4). Div #57: there is no season/round/driver/session CATALOG
// endpoint, so Season is a static known set and Driver is populated from GET /drivers; Round (event)
// and Session are bounded inputs; the "laps available" count has no endpoint and is rendered as an
// explicit unavailable state, never guessed. Submit drives the real async ingestion job (div #54).
const SEASONS = [2024, 2023, 2022] as const;
const SESSIONS = ['R', 'Q', 'FP1', 'FP2', 'FP3'] as const;

const fieldClass =
  'rounded border border-border bg-background px-2 py-1 font-mono text-xs text-text-primary';

export function FastF1Tab() {
  const [drivers, setDrivers] = useState<DriverSummary[]>([]);
  const [year, setYear] = useState<number>(SEASONS[0]);
  const [event, setEvent] = useState<string>('');
  const [session, setSession] = useState<string>(SESSIONS[0]);
  const [driver, setDriver] = useState<string>('');
  const { state, submit } = useIngestionJob();

  useEffect(() => {
    let alive = true;
    void getDrivers().then((r) => {
      if (alive && r.ok) setDrivers(r.data);
    });
    return () => {
      alive = false;
    };
  }, []);

  const phase: PipelinePhase = state.phase === 'idle' ? 'IDLE' : (state.status as PipelinePhase);
  const running = state.phase === 'enqueuing' || state.phase === 'running';

  return (
    <div data-testid="fastf1-tab">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] text-text-secondary">Season</span>
          <select
            data-testid="season-select"
            aria-label="Season"
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            className={fieldClass}
          >
            {SEASONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] text-text-secondary">Round (event)</span>
          <input
            data-testid="event-input"
            aria-label="Round (event)"
            value={event}
            onChange={(e) => setEvent(e.target.value)}
            placeholder="e.g. Bahrain"
            className={fieldClass}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] text-text-secondary">Driver</span>
          <select
            data-testid="driver-select"
            aria-label="Driver"
            value={driver}
            onChange={(e) => setDriver(e.target.value)}
            className={fieldClass}
          >
            <option value="">All drivers</option>
            {drivers.map((d) => (
              <option key={d.code} value={d.code}>
                {d.code}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] text-text-secondary">Session</span>
          <select
            data-testid="session-select"
            aria-label="Session"
            value={session}
            onChange={(e) => setSession(e.target.value)}
            className={fieldClass}
          >
            {SESSIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <Button
          active
          disabled={!event.trim() || running}
          onClick={() => void submit({ year, event: event.trim(), session })}
          data-testid="ingest-submit"
        >
          Ingest session
        </Button>
      </div>

      {/* Div #57: no endpoint yields the expected clean-lap count — shown honestly, never guessed. */}
      <div className="mt-3">
        <EmptyState
          message="Laps available: not exposed by the API (no session-catalog endpoint)."
          className="py-3"
        />
      </div>

      {state.phase !== 'idle' ? (
        <div className="mt-4">
          <PipelineProgress status={phase} progress={state.progress} errorDetail={state.errorDetail} />
        </div>
      ) : null}

      {state.phase === 'error' && state.error ? (
        <div className="mt-3">
          <ErrorCard code={state.error.code} detail={state.error.detail} />
        </div>
      ) : null}

      {state.phase === 'complete' && state.result ? (
        <div className="mt-3" data-testid="ingest-result">
          <h3 className="mb-1 font-mono text-[11px] uppercase text-text-secondary">Ingestion report</h3>
          <pre className="overflow-x-auto rounded border border-border bg-background p-2 font-mono text-[11px] text-text-primary">
            {JSON.stringify(state.result, null, 2)}
          </pre>
        </div>
      ) : null}
    </div>
  );
}
