'use client';

import { EmptyState } from '@/components/ds/EmptyState';
import { FEATURE_LABELS } from '@/lib/design/features';

// Most-distinctive features (master §10.5): top 5 by |SHAP delta| as horizontal bars from zero,
// hover = plain-English meaning. The redesign upgrades this from the /compare per-DOMAIN delta
// (the old div #61 limitation) to a REAL per-FEATURE delta: driver1.shap_signed −
// driver2.shap_signed, both served by /drivers/{code}/fingerprint. When either fingerprint's
// signed vector is unavailable the panel falls back to the served per-domain delta, labelled
// honestly — never an invented feature ranking.
const TOP_N = 5;

type Props = {
  featureDelta: Record<string, number> | null; // per-feature signed SHAP delta; null → fallback
  domainDelta: Record<string, number>; // /compare delta_shap (per-domain), the fallback
  driver1: string;
  driver2: string;
};

function rank(delta: Record<string, number>): string[] {
  return Object.keys(delta)
    .filter((k) => Number.isFinite(delta[k]))
    .sort((a, b) => Math.abs(delta[b] ?? 0) - Math.abs(delta[a] ?? 0) || a.localeCompare(b))
    .slice(0, TOP_N);
}

export function DistinctiveFeatures({ featureDelta, domainDelta, driver1, driver2 }: Props) {
  const perFeature = featureDelta != null && Object.keys(featureDelta).length > 0;
  const delta = perFeature ? featureDelta : domainDelta;
  const keys = rank(delta);

  if (keys.length === 0) {
    return <EmptyState message="No distinguishing signals returned." className="py-3" />;
  }

  const maxAbs = Math.max(...keys.map((k) => Math.abs(delta[k] ?? 0)), 1e-9);

  return (
    <div data-testid="distinctive-features" data-mode={perFeature ? 'feature' : 'domain'}>
      {!perFeature ? (
        <p className="mb-2 font-mono text-[9px] text-alert-ink" data-testid="domain-fallback-note">
          Signed per-feature SHAP unavailable for this pair — showing the per-domain delta.
        </p>
      ) : null}
      <ul className="space-y-2.5">
        {keys.map((k) => {
          const v = delta[k] ?? 0;
          const pct = (Math.abs(v) / maxAbs) * 100;
          const positive = v >= 0;
          const label = perFeature ? (FEATURE_LABELS[k] ?? k) : k;
          const lead = positive ? driver1 : driver2;
          const meaning = perFeature
            ? `${lead}'s identification leans more on ${label} (${k}, Δ ${v >= 0 ? '+' : ''}${v.toFixed(2)})`
            : `${lead} higher on ${k} (${v >= 0 ? '+' : ''}${v.toFixed(2)})`;
          return (
            <li key={k} data-testid="distinctive-row" data-key={k} data-sign={positive ? 'positive' : 'negative'} title={meaning}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-[11px] text-text-primary">
                  {label}
                  {perFeature ? <span className="ml-1.5 text-[9px] text-text-secondary">{k}</span> : null}
                </span>
                <span className="font-mono text-[10px] tabular-nums text-text-secondary">
                  {positive ? '+' : ''}
                  {v.toFixed(2)}
                </span>
              </div>
              <svg viewBox="0 0 100 4" preserveAspectRatio="none" className="mt-1 h-1 w-full" aria-hidden="true">
                <rect x={0} y={0} width={100} height={4} className="fill-border" />
                <rect
                  x={0}
                  y={0}
                  width={pct}
                  height={4}
                  className={positive ? 'fill-delta-up' : 'fill-delta-down'}
                  data-testid="distinctive-fill"
                />
              </svg>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 font-mono text-[9px] text-text-secondary" data-testid="distinctive-legend">
        red = leans toward {driver1} · blue = toward {driver2} · Δ of mean signed SHAP
      </p>
    </div>
  );
}
