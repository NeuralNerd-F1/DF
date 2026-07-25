'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import { getIngestStatus, postIngestSession } from '@/lib/api/endpoints';
import type { ApiErr } from '@/lib/api/errors';
import type { IngestSessionRequest, IngestStatusResponse } from '@/lib/api/types';
import { DURATION } from '@/lib/design/animation';

// The FastF1 ingestion job (master §10.4): enqueue via POST /ingest/session (202) then poll
// GET /ingest/status until COMPLETE/FAILED. Div #54/#55: ingestion is async and writes to the DB —
// it returns NO prediction, only a counts/quality report. The poll timer is cleared on every
// terminal status AND on unmount (no leak, no API hammering — see the plan's polling risk).
export type JobPhase = 'idle' | 'enqueuing' | 'running' | 'complete' | 'failed' | 'error';

export type JobState = {
  phase: JobPhase;
  status: IngestStatusResponse['status'] | 'IDLE';
  progress: number;
  result: Record<string, unknown> | null;
  errorDetail: string | null;
  error: ApiErr | null;
};

const INITIAL: JobState = {
  phase: 'idle',
  status: 'IDLE',
  progress: 0,
  result: null,
  errorDetail: null,
  error: null,
};

export function useIngestionJob(pollMs: number = DURATION.ingestPoll) {
  const [state, setState] = useState<JobState>(INITIAL);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  const submit = useCallback(
    async (body: IngestSessionRequest) => {
      stop();
      setSessionId(null);
      setState({ ...INITIAL, phase: 'enqueuing', status: 'PENDING' });
      const r = await postIngestSession(body);
      if (!r.ok) {
        setState({ ...INITIAL, phase: 'error', error: r });
        return;
      }
      setState({ ...INITIAL, phase: 'running', status: 'PENDING' });
      setSessionId(r.data.session_id);
    },
    [stop],
  );

  useEffect(() => {
    if (!sessionId) return;
    let alive = true;

    const poll = async () => {
      const r = await getIngestStatus(sessionId);
      if (!alive) return;
      if (!r.ok) {
        setState((s) => ({ ...s, phase: 'error', error: r }));
        stop();
        return;
      }
      const d = r.data;
      const terminal = d.status === 'COMPLETE' || d.status === 'FAILED';
      setState((s) => ({
        ...s,
        phase: d.status === 'COMPLETE' ? 'complete' : d.status === 'FAILED' ? 'failed' : 'running',
        status: d.status,
        progress: d.progress,
        result: (d.result as Record<string, unknown> | null) ?? s.result,
        errorDetail: d.error_detail ?? s.errorDetail,
      }));
      if (terminal) stop();
    };

    void poll(); // immediate first poll, then on the interval
    timer.current = setInterval(() => void poll(), pollMs);
    return () => {
      alive = false;
      stop();
    };
  }, [sessionId, pollMs, stop]);

  // Cleanup on unmount.
  useEffect(() => stop, [stop]);

  return { state, submit };
}
