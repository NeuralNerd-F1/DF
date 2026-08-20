import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import { RADAR_DOMAINS, type RadarDomain } from '@/lib/api/types';

// Screen 2 style-profile card: the large named-axis radar plus a per-domain score list. Both render
// the SAME grid-relative values (share of the grid's strongest driver per domain, computed by the
// page from the full /drivers payload) — the radar shows the shape, the list makes each axis
// legible as a number. 100 = the grid's strongest on that axis, by construction.
export function StyleProfileCard({
  radar,
  colour,
  code,
}: {
  radar: Record<RadarDomain, number> | undefined;
  colour: string;
  code: string;
}) {
  return (
    <section className="rounded-md border border-border bg-card p-4" data-testid="style-profile-card">
      <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        Style profile
      </h2>
      <p className="mt-1 font-mono text-[10px] leading-snug text-text-secondary">
        Where the classifier&apos;s attention concentrates for {code}, as mean |SHAP| per domain,
        scaled to the grid maximum on each axis.
      </p>
      <div className="mt-2 flex justify-center">
        <FingerprintRadar
          values={radar ?? {}}
          size={250}
          showLabels
          colour={colour}
          ariaLabel={`${code} style profile, axes scaled to grid maximum`}
        />
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 border-t border-border pt-3">
        {RADAR_DOMAINS.map((domain) => {
          const value = radar?.[domain];
          const score = typeof value === 'number' ? Math.round(value * 100) : null;
          return (
            <div key={domain} className="flex items-center gap-2" data-testid={`domain-score-${domain}`}>
              <dt className="w-20 shrink-0 font-mono text-[9px] uppercase tracking-[0.14em] text-text-secondary">
                {domain}
              </dt>
              <svg viewBox="0 0 100 4" preserveAspectRatio="none" className="h-1 flex-1" aria-hidden="true">
                <rect width={100} height={4} className="fill-border" />
                {score !== null ? <rect width={score} height={4} fill={colour} /> : null}
              </svg>
              <dd className="w-7 text-right font-mono text-[11px] tabular-nums text-text-primary">
                {score !== null ? score : '—'}
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
