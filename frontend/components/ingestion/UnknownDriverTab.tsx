'use client';

import { useState } from 'react';

import { Button } from '@/components/ds/Button';
import { ConfidenceBadge } from '@/components/ds/ConfidenceBadge';
import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { messageFor } from '@/lib/api/errors';
import type { ApiResult } from '@/lib/api/errors';
import { postUnknownFingerprint } from '@/lib/api/endpoints';
import type { ConfidenceState, UnknownFingerprintResponse } from '@/lib/api/types';
import { CIRCUITS, DEFAULT_CIRCUIT } from '@/lib/design/circuits';

import { NearestNeighbours } from './NearestNeighbours';
import { RiskFlags } from './RiskFlags';
import { SAMPLE_LAPS_JSON } from './sampleVector';

// Unknown Driver tab (master §10.4). Real: NN matches, risk flags, data-quality, and — div #60 —
// the confidence_state + warnings (this is the honest live home of LOW_CONFIDENCE / OOD /
// SPARSE_PROJECTION for projected drivers). Div #59: radar_values/style_description come back EMPTY
// from the route and there is no SSE, so the axis-by-axis radar is an explicit EmptyState and the
// style-space embed (Screen 5) is a deferred F18 state — never a zeroed/fake radar.
function parseLaps(text: string): { laps: Array<Record<string, number | null>>; error: string | null } {
  try {
    const raw = JSON.parse(text) as unknown;
    if (!Array.isArray(raw)) return { laps: [], error: 'Laps must be a JSON array of feature objects.' };
    return { laps: raw as Array<Record<string, number | null>>, error: null };
  } catch {
    return { laps: [], error: 'Invalid JSON.' };
  }
}

const fieldClass =
  'rounded border border-border bg-background px-2 py-1 font-mono text-xs text-text-primary';

export function UnknownDriverTab() {
  const [text, setText] = useState<string>(SAMPLE_LAPS_JSON);
  const [circuit, setCircuit] = useState<string>(DEFAULT_CIRCUIT);
  const [loading, setLoading] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResult<UnknownFingerprintResponse> | null>(null);

  async function run() {
    const { laps, error } = parseLaps(text);
    setParseError(error);
    if (error) return;
    setLoading(true);
    const r = await postUnknownFingerprint({ laps, circuit });
    setResult(r);
    setLoading(false);
  }

  return (
    <div data-testid="unknown-driver-tab">
      <label className="mb-1 block font-mono text-[11px] text-text-secondary">
        Unlabelled laps (synthetic 6-lap example — edit or paste feature rows)
      </label>
      <textarea
        data-testid="laps-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        rows={6}
        className="w-full rounded border border-border bg-background p-2 font-mono text-[11px] text-text-primary"
      />
      <div className="mt-2 flex items-center gap-3">
        <select
          data-testid="unknown-circuit"
          aria-label="Circuit"
          value={circuit}
          onChange={(e) => setCircuit(e.target.value)}
          className={fieldClass}
        >
          {CIRCUITS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <Button active onClick={() => void run()} disabled={loading} data-testid="unknown-submit">
          Project fingerprint
        </Button>
      </div>

      {parseError ? (
        <p data-testid="parse-error" className="mt-2 font-mono text-[11px] text-accent">
          {parseError}
        </p>
      ) : null}

      <div className="mt-4">
        {loading ? (
          <Skeleton className="h-24" label="Projecting fingerprint" />
        ) : result && !result.ok ? (
          <ErrorCard code={result.code} detail={result.detail} onRetry={() => void run()} />
        ) : result && result.ok ? (
          <UnknownResult data={result.data} />
        ) : null}
      </div>
    </div>
  );
}

function UnknownResult({ data }: { data: UnknownFingerprintResponse }) {
  const warnings = data.warnings ?? [];
  return (
    <div data-testid="unknown-result" className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <ConfidenceBadge state={(data.confidence_state as ConfidenceState | null) ?? null} />
        {warnings.map((w) => (
          <span
            key={w}
            data-testid="unknown-warning"
            data-code={w}
            className="rounded border border-alert-ink px-1.5 py-0.5 font-mono text-[10px] text-alert-ink"
          >
            {messageFor(w)}
          </span>
        ))}
      </div>

      {/* Div #59: the route returns radar_values = {} — render honestly, never a zeroed radar. */}
      <div>
        <h3 className="mb-1 font-mono text-[11px] uppercase text-text-secondary">Fingerprint radar</h3>
        <EmptyState message="Unknown-driver radar not populated by the API (radar_values empty)." className="py-4" />
      </div>

      <div>
        <h3 className="mb-1 font-mono text-[11px] uppercase text-text-secondary">Nearest drivers</h3>
        {data.nearest_neighbours && data.nearest_neighbours.length > 0 ? (
          <NearestNeighbours neighbours={data.nearest_neighbours} />
        ) : (
          <EmptyState message="No nearest-neighbour matches returned." className="py-3" />
        )}
      </div>

      {data.risk_flags && data.risk_flags.length > 0 ? (
        <div>
          <h3 className="mb-1 font-mono text-[11px] uppercase text-text-secondary">Risk flags</h3>
          <RiskFlags flags={data.risk_flags} />
        </div>
      ) : null}

      <dl data-testid="data-quality" className="flex gap-4 font-mono text-[11px] text-text-secondary">
        <div>
          <dt className="inline">laps used</dt> <dd className="inline text-text-primary">{data.data_quality.n_laps_used}</dd>
        </div>
        <div>
          <dt className="inline">rejected</dt> <dd className="inline text-text-primary">{data.data_quality.n_laps_rejected}</dd>
        </div>
      </dl>

      {/* Div #59: the style-space embed is Screen 5 (F18) — deferred, not faked. */}
      <EmptyState message="Style Space projection (Screen 5) arrives in F18." className="py-3" />
    </div>
  );
}
