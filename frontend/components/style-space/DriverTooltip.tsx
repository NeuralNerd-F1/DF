'use client';

import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import type { Neighbour, StyleNode } from '@/lib/style-space/styleSpace';

// Hover tooltip (master §10.6): driver code, team, season, nearest neighbours, and a mini radar.
// `season` is "—" (div #64: /stylespace never populates it). Neighbours are client-computed (#66);
// the mini radar reuses FingerprintRadar with the joined radar_values (absent → omitted, honest).
export function DriverTooltip({ node, neighbours }: { node: StyleNode; neighbours: Neighbour[] }) {
  return (
    <div
      data-testid="driver-tooltip"
      className="w-48 rounded-lg border border-border bg-card p-3 shadow-lg"
    >
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-sm text-text-primary" data-testid="tooltip-code">
          {node.code}
        </span>
        <span className="font-mono text-[10px] text-text-secondary">{node.team ?? '—'}</span>
      </div>
      <div className="font-mono text-[10px] text-text-secondary" data-testid="tooltip-season">
        season: —
      </div>
      {node.radar ? (
        <div className="mt-2 flex justify-center">
          <FingerprintRadar values={node.radar} size={120} showLabels={false} animate={false} />
        </div>
      ) : null}
      {neighbours.length > 0 ? (
        <ul data-testid="tooltip-neighbours" className="mt-2 space-y-0.5">
          {neighbours.map((n) => (
            <li key={n.code} className="flex justify-between font-mono text-[10px] text-text-secondary">
              <span>{n.code}</span>
              <span>{n.similarity.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
