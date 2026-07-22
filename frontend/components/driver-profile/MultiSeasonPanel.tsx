import type { StabilityResponse } from '@/lib/api/types';

// The multi-season data that IS exposed (master §10.3 overlay): seasons available, stability score,
// and the pairwise Jensen–Shannon divergence. Per-season radar SWITCHING/overlay needs a per-season
// fingerprint endpoint the API does not expose, so that is surfaced honestly, never faked.
export function MultiSeasonPanel({
  stability,
}: {
  stability: StabilityResponse | null | undefined;
}) {
  if (!stability) return null;
  const js = Object.entries(stability.pairwise_js ?? {});
  return (
    <section data-testid="multiseason-panel">
      <h3 className="mb-2 font-mono text-xs text-text-secondary">Multi-season</h3>
      <dl className="grid grid-cols-2 gap-1 font-mono text-[11px]">
        <div>
          <dt className="text-text-secondary">Seasons</dt>
          <dd className="text-text-primary">
            {stability.seasons_available}
            {stability.partial ? ' (partial)' : ''}
          </dd>
        </div>
        <div>
          <dt className="text-text-secondary">Stability</dt>
          <dd className="text-text-primary">
            {stability.stability_score != null ? stability.stability_score.toFixed(2) : '—'}
          </dd>
        </div>
      </dl>
      {js.length > 0 ? (
        <div className="mt-2" data-testid="js-divergence">
          <div className="text-[10px] text-text-secondary">JS divergence</div>
          <ul className="font-mono text-[10px] text-text-primary">
            {js.map(([pair, val]) => (
              <li key={pair}>
                <span className="text-text-secondary">{pair}</span> {val.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="mt-1 text-[10px] text-text-secondary">
          Per-season overlay needs a per-season endpoint (not yet exposed).
        </p>
      )}
    </section>
  );
}
