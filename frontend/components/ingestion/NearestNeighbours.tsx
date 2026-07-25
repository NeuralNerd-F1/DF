'use client';

import type { NeighbourSchema } from '@/lib/api/types';

// Nearest-neighbour matches (master §10.4 Unknown Driver tab): the top-3 known drivers with their
// similarity shown as filled progress bars. Real data from POST /fingerprint/unknown.
function clamp01(v: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

export function NearestNeighbours({ neighbours }: { neighbours: NeighbourSchema[] }) {
  const top = neighbours.slice(0, 3);
  if (top.length === 0) return null;
  return (
    <ul data-testid="nearest-neighbours" className="space-y-2">
      {top.map((n) => {
        const pct = clamp01(n.cosine_similarity) * 100;
        return (
          <li key={n.driver} data-testid="nn-row" data-driver={n.driver}>
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-xs text-text-primary">{n.driver}</span>
              <span className="font-mono text-[10px] text-text-secondary">
                {n.cosine_similarity.toFixed(2)}
              </span>
            </div>
            <svg viewBox="0 0 100 4" preserveAspectRatio="none" className="mt-1 h-1 w-full" aria-hidden="true">
              <rect x={0} y={0} width={100} height={4} className="fill-border" />
              <rect x={0} y={0} width={pct} height={4} className="fill-accent" data-testid="nn-fill" />
            </svg>
          </li>
        );
      })}
    </ul>
  );
}
