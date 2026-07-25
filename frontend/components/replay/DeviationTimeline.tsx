'use client';

import type { ReplayLap } from '@/lib/api/types';
import { ALERT_FILL_CLASS, deviationMax } from '@/lib/replay/replay';

// The tyre-degradation timeline (master §10.7, div #73). A left-building line of deviation_score per
// replayed lap (deterministic SVG); ALERT laps highlighted/pulsed red; null scores render as GAPS
// (never 0). The master's dashed τ threshold line is DEFERRED — τ is not exposed per lap — rendered
// as a labelled note, never a fabricated line. Named seam: ReplayLap.threshold.
const W = 300;
const H = 80;
const PAD = 6;

type Props = { laps: readonly ReplayLap[]; reachedIndex: number; className?: string };

export function DeviationTimeline({ laps, reachedIndex, className = '' }: Props) {
  const max = deviationMax(laps);
  const n = laps.length;
  const xOf = (i: number): number => (n <= 1 ? W / 2 : PAD + (i / (n - 1)) * (W - 2 * PAD));
  const yOf = (s: number): number => H - PAD - (Math.max(0, Math.min(max, s)) / max) * (H - 2 * PAD);

  // Only laps up to the reached frontier are shown; segments break at nulls (gaps).
  const shown = laps.slice(0, Math.min(reachedIndex, n - 1) + 1);
  const segments: string[] = [];
  let cur: string[] = [];
  shown.forEach((lap, i) => {
    const s = lap.deviation_score;
    if (typeof s === 'number' && Number.isFinite(s)) {
      cur.push(`${xOf(i).toFixed(2)},${yOf(s).toFixed(2)}`);
    } else if (cur.length) {
      segments.push(cur.join(' '));
      cur = [];
    }
  });
  if (cur.length) segments.push(cur.join(' '));

  return (
    <div className={className} data-testid="deviation-timeline">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label="deviation timeline">
        {segments.map((pts, i) => (
          <polyline key={i} points={pts} fill="none" className="stroke-accent" strokeWidth={1.5} />
        ))}
        {shown.map((lap, i) => {
          const s = lap.deviation_score;
          if (typeof s !== 'number' || !Number.isFinite(s)) return null;
          const isAlert = lap.alert_state === 'ALERT';
          return (
            <circle
              key={i}
              cx={xOf(i)}
              cy={yOf(s)}
              r={isAlert ? 3 : 1.5}
              className={isAlert ? `${ALERT_FILL_CLASS.alert} animate-pulse` : 'fill-text-secondary'}
              data-testid={isAlert ? 'timeline-alert-point' : 'timeline-point'}
            />
          );
        })}
      </svg>
      <p className="mt-1 font-mono text-[9px] text-text-secondary" data-testid="timeline-threshold-note">
        Threshold line unavailable (τ not exposed per lap)
      </p>
    </div>
  );
}
