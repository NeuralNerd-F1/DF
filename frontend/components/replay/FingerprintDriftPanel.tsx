'use client';

import { ConfidenceBadge } from '@/components/ds/ConfidenceBadge';
import type { ReplayLap } from '@/lib/api/types';
import { driftTrail, narrowSnapshot } from '@/lib/replay/replay';

// The evolving-fingerprint panel (master §10.7 "the fingerprint evolving in real time"). Each lap's
// REAL F08 snapshot (the projection recomputed over the growing prefix) is plotted as a trail in
// the style-space 2D plane — older laps fade, the reached lap is emphasised — with the per-lap
// confidence_state alongside. The 6-domain radar stays unexposed per lap (div #75: the system
// computes none — named seam ReplayLap.radar_values); this panel renders only measured coordinates,
// never a radar fabricated from them.
const W = 260;
const H = 160;
const PAD = 16;

type Props = { laps: readonly ReplayLap[]; reachedIndex: number; colour?: string };

export function FingerprintDriftPanel({ laps, reachedIndex, colour = '#E10600' }: Props) {
  // Bounds over the FULL replay so the frame holds still during playback (no swimming axes).
  const all = driftTrail(laps, laps.length - 1);
  const shown = driftTrail(laps, reachedIndex);
  const current = shown[shown.length - 1] ?? null;
  const idx = Math.min(reachedIndex, laps.length - 1);
  const confidence = narrowSnapshot(laps[idx]?.fingerprint_snapshot)?.confidence_state ?? null;

  if (all.length === 0) {
    return (
      <div data-testid="fingerprint-drift">
        <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
          Fingerprint evolution
        </p>
        <p className="mt-2 font-mono text-[10px] text-text-secondary" data-testid="drift-empty-note">
          No fingerprint snapshots in this replay — the style-space projection was unavailable while
          it ran.
        </p>
      </div>
    );
  }

  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const sx = (x: number): number =>
    x1 === x0 ? W / 2 : PAD + ((x - x0) / (x1 - x0)) * (W - 2 * PAD);
  const sy = (y: number): number =>
    y1 === y0 ? H / 2 : H - PAD - ((y - y0) / (y1 - y0)) * (H - 2 * PAD);

  return (
    <div data-testid="fingerprint-drift">
      <div className="flex items-baseline justify-between">
        <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
          Fingerprint evolution
        </p>
        {confidence ? (
          <span className="flex items-center gap-2" data-testid="drift-confidence">
            <span className="font-mono text-[9px] text-text-secondary">{confidence}</span>
            <ConfidenceBadge state={confidence} />
          </span>
        ) : null}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 w-full" aria-label="fingerprint drift trail">
        {shown.length > 1 ? (
          <polyline
            points={shown.map((p) => `${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`).join(' ')}
            fill="none"
            stroke={colour}
            strokeOpacity={0.35}
            strokeWidth={1}
            data-testid="drift-trail"
          />
        ) : null}
        {shown.map((p) => {
          const isCurrent = current != null && p.index === current.index;
          // Older points fade towards the tail; the reached lap is solid and larger.
          const age = current != null ? current.index - p.index : 0;
          const opacity = isCurrent ? 1 : Math.max(0.15, 0.6 - age * 0.04);
          return (
            <circle
              key={p.index}
              cx={sx(p.x)}
              cy={sy(p.y)}
              r={isCurrent ? 4 : 1.8}
              fill={colour}
              fillOpacity={opacity}
              data-testid={isCurrent ? 'drift-current' : 'drift-point'}
            />
          );
        })}
      </svg>
      <p className="mt-1 font-mono text-[9px] leading-relaxed text-text-secondary" data-testid="drift-caption">
        The F08 style-space projection recomputed as each lap arrives — the driver&apos;s fingerprint
        as the monitor sees it mid-stint. Projection axes carry no physical units; the per-lap
        6-domain radar is not exposed by the system.
      </p>
    </div>
  );
}
