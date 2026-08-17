'use client';

import { useEffect, useState } from 'react';

import { getHealth } from '@/lib/api/endpoints';

// What the header dot claims, resolved from the REAL /health response — never assumed. Three
// honest states: every model loaded, some loaded, or the API itself unreachable. The initial
// render (both server and client, so hydration agrees) is the probing state.
type Status =
  | { kind: 'probing' }
  | { kind: 'active'; loaded: number; total: number }
  | { kind: 'partial'; loaded: number; total: number }
  | { kind: 'offline' };

export function SystemStatus() {
  const [status, setStatus] = useState<Status>({ kind: 'probing' });

  useEffect(() => {
    let alive = true;
    void getHealth().then((r) => {
      if (!alive) return;
      if (!r.ok) {
        setStatus({ kind: 'offline' });
        return;
      }
      const flags = Object.values(r.data.models_loaded ?? {});
      const loaded = flags.filter(Boolean).length;
      setStatus(
        flags.length > 0 && loaded === flags.length
          ? { kind: 'active', loaded, total: flags.length }
          : { kind: 'partial', loaded, total: flags.length },
      );
    });
    return () => {
      alive = false;
    };
  }, []);

  const { dot, text, tone } =
    status.kind === 'active'
      ? { dot: 'bg-accent', text: 'Model active', tone: 'text-accent' }
      : status.kind === 'partial'
        ? {
            dot: 'bg-alert-ink',
            text: `${status.loaded}/${status.total} models`,
            tone: 'text-alert-ink',
          }
        : status.kind === 'offline'
          ? { dot: 'bg-unavailable', text: 'API offline', tone: 'text-unavailable' }
          : { dot: 'bg-border', text: 'Probing…', tone: 'text-text-secondary' };

  return (
    <span
      className={`flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] ${tone}`}
      data-testid="system-status"
      data-status={status.kind}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} aria-hidden="true" />
      {text}
    </span>
  );
}
