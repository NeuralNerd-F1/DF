'use client';

import { EmptyState } from '@/components/ds/EmptyState';
import type { CompareResponse } from '@/lib/api/types';

// Most-distinctive features (master §10.5): the domains that differ most between the two drivers, as
// horizontal bars from zero. Div #61: the backend delta is per-DOMAIN (6 domains), not per-feature,
// so this is "top domains", labelled honestly. Positive (driver-1 higher) red, negative blue.
function meaning(domain: string, delta: number): string {
  const lead = delta >= 0 ? 'Driver 1' : 'Driver 2';
  return `${lead} higher on ${domain} (${delta >= 0 ? '+' : ''}${delta.toFixed(2)})`;
}

export function DistinctiveFeatures({ compare }: { compare: CompareResponse }) {
  const delta = compare.delta_shap ?? {};
  const ranked =
    compare.most_distinctive_features && compare.most_distinctive_features.length > 0
      ? compare.most_distinctive_features
      : Object.keys(delta).sort((a, b) => Math.abs(delta[b] ?? 0) - Math.abs(delta[a] ?? 0));
  const domains = ranked.slice(0, 5);

  if (domains.length === 0) {
    return <EmptyState message="No distinguishing domains returned." className="py-3" />;
  }

  const maxAbs = Math.max(...domains.map((d) => Math.abs(delta[d] ?? 0)), 1e-9);

  return (
    <ul data-testid="distinctive-features" className="space-y-2">
      {domains.map((d) => {
        const v = delta[d] ?? 0;
        const pct = (Math.abs(v) / maxAbs) * 100;
        const positive = v >= 0;
        return (
          <li
            key={d}
            data-testid="distinctive-row"
            data-domain={d}
            data-sign={positive ? 'positive' : 'negative'}
            title={meaning(d, v)}
          >
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-xs capitalize text-text-primary">{d}</span>
              <span className="font-mono text-[10px] text-text-secondary">
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
  );
}
