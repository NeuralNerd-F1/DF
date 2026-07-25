'use client';

import { useState } from 'react';

import { Button } from '@/components/ds/Button';
import { ConfidenceBadge } from '@/components/ds/ConfidenceBadge';
import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { NearestNeighbours } from '@/components/ingestion/NearestNeighbours';
import { SAMPLE_LAPS_JSON } from '@/components/ingestion/sampleVector';
import { postUnknownFingerprint } from '@/lib/api/endpoints';
import { messageFor, type ApiResult } from '@/lib/api/errors';
import type { ConfidenceState, UnknownFingerprintResponse } from '@/lib/api/types';
import { CIRCUITS, DEFAULT_CIRCUIT } from '@/lib/design/circuits';

// Unknown-driver projection (master §10.6). Reuses F17's postUnknownFingerprint + NearestNeighbours.
// On success it lifts the result to the parent (which places the 3D white sphere from the 5D
// fingerprint_vector). Div #67: when fingerprint_vector is null (PROJECTION_FALLBACK) the panel shows
// "position unavailable — matches shown", never a fabricated position.
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

export function UnknownProjectionPanel({
  onProjected,
}: {
  onProjected: (r: UnknownFingerprintResponse | null) => void;
}) {
  const [text, setText] = useState(SAMPLE_LAPS_JSON);
  const [circuit, setCircuit] = useState(DEFAULT_CIRCUIT);
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
    onProjected(r.ok ? r.data : null);
  }

  return (
    <details data-testid="unknown-projection">
      <summary className="cursor-pointer font-mono text-xs text-text-secondary">
        Project an unknown driver
      </summary>
      <textarea
        data-testid="unknown-laps"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        spellCheck={false}
        className="mt-2 w-full rounded border border-border bg-background p-2 font-mono text-[11px] text-text-primary"
      />
      <div className="mt-2 flex items-center gap-2">
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
        <Button active onClick={() => void run()} disabled={loading} data-testid="unknown-project-submit">
          Project
        </Button>
      </div>
      {parseError ? (
        <p data-testid="parse-error" className="mt-2 font-mono text-[11px] text-accent">
          {parseError}
        </p>
      ) : null}
      <div className="mt-3">
        {loading ? (
          <Skeleton className="h-16" label="Projecting" />
        ) : result && !result.ok ? (
          <ErrorCard code={result.code} detail={result.detail} onRetry={() => void run()} />
        ) : result && result.ok ? (
          <UnknownResult data={result.data} />
        ) : null}
      </div>
    </details>
  );
}

function UnknownResult({ data }: { data: UnknownFingerprintResponse }) {
  const placed = data.fingerprint_vector != null && data.fingerprint_vector.length > 0;
  const warnings = data.warnings ?? [];
  return (
    <div data-testid="unknown-projection-result">
      <div className="flex flex-wrap items-center gap-2">
        <ConfidenceBadge state={(data.confidence_state as ConfidenceState | null) ?? null} />
        {warnings.map((w) => (
          <span
            key={w}
            data-testid="unknown-warning"
            data-code={w}
            className="rounded border border-confidence-medium px-1.5 py-0.5 font-mono text-[10px] text-confidence-medium"
          >
            {messageFor(w)}
          </span>
        ))}
      </div>
      {!placed ? (
        <p data-testid="unknown-unplaced" className="mt-2 font-mono text-[11px] text-text-secondary">
          Position unavailable — nearest matches shown.
        </p>
      ) : null}
      <div className="mt-2">
        {data.nearest_neighbours && data.nearest_neighbours.length > 0 ? (
          <NearestNeighbours neighbours={data.nearest_neighbours} />
        ) : (
          <EmptyState message="No matches returned." className="py-2" />
        )}
      </div>
    </div>
  );
}
