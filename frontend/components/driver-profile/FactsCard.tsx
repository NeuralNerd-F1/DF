import type { DriverSummary, StabilityResponse } from '@/lib/api/types';

function Row({
  label,
  value,
  hint,
  missing,
}: {
  label: string;
  value: string;
  hint: string;
  missing?: boolean;
}) {
  return (
    <div title={hint}>
      <dt className="font-mono text-[8.5px] uppercase tracking-[0.18em] text-text-secondary">
        {label}
      </dt>
      <dd
        className={`mt-0.5 font-mono text-[15px] tabular-nums ${
          missing ? 'text-text-secondary' : 'text-text-primary'
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

// Screen 2 facts card — the dataset + evaluation numbers behind everything else on the page.
// Sources: the profiles artefact (via /drivers) and the stored multi-season stability
// (via /fingerprint). Dash = not measured; a partial stability window says so instead of
// presenting one season as a settled score.
export function FactsCard({
  driver,
  stability,
}: {
  driver: DriverSummary;
  stability: StabilityResponse | null | undefined;
}) {
  const p = driver.profile ?? null;
  const seasons =
    p && p.seasons && p.seasons.length > 0
      ? p.seasons.length === 1
        ? String(p.seasons[0])
        : `${Math.min(...p.seasons)} to ${Math.max(...p.seasons)}`
      : null;

  return (
    <section className="rounded-md border border-border bg-card p-4" data-testid="facts-card">
      <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        Dataset &amp; evaluation
      </h2>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3">
        <Row
          label="Laps analysed"
          value={p ? p.laps_analysed.toLocaleString('en-GB') : '—'}
          hint="Clean laps of this driver in the training matrix"
          missing={!p}
        />
        <Row
          label="Consistency"
          value={p?.lap_time_consistency_s != null ? `±${p.lap_time_consistency_s.toFixed(2)}s` : '—'}
          hint="Median within-session lap-time spread over clean laps. Lower means more repeatable."
          missing={p?.lap_time_consistency_s == null}
        />
        <Row
          label="ID rate"
          value={p?.identification_f1 != null ? `${Math.round(p.identification_f1 * 100)}%` : '—'}
          hint="How reliably the classifier identifies this driver on circuits it never trained on (held-out F1)"
          missing={p?.identification_f1 == null}
        />
        <Row
          label="Seasons"
          value={seasons ?? '—'}
          hint="Seasons this driver appears in within the training matrix"
          missing={!seasons}
        />
        <Row
          label="Circuits"
          value={p ? String(p.n_circuits) : '—'}
          hint="Distinct circuits in this driver's training laps (held-out circuits excluded)"
          missing={!p}
        />
        <Row
          label="Fingerprint stability"
          value={
            stability?.stability_score != null
              ? stability.stability_score.toFixed(2)
              : stability
                ? `${stability.seasons_available} season${stability.seasons_available === 1 ? '' : 's'}${stability.partial ? ' (partial)' : ''}`
                : '—'
          }
          hint="Cross-season fingerprint stability; needs 3+ stored seasons before a score exists"
          missing={stability?.stability_score == null}
        />
      </dl>
    </section>
  );
}
