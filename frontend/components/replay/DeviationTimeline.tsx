'use client';

import type { ReplayLap } from '@/lib/api/types';
import { ALERT_FILL_CLASS, gaugeMax, stintThreshold } from '@/lib/replay/replay';

// The tyre-degradation timeline (master §10.7). A left-building line of deviation_score per
// replayed lap (deterministic SVG); ALERT laps highlighted/pulsed red; null scores render as GAPS
// (never 0). τ ships per lap now (div #73 closed): the master's dashed red threshold line is REAL —
// the y-scale anchors to τ with headroom (gaugeMax) so the line always sits inside the chart.
const W = 300;
const H = 80;
const PAD = 6;

type Props = {
  laps: readonly ReplayLap[];
  reachedIndex: number;
  className?: string;
  // compact: grid cells — SVG only, no caption row (the legend text doesn't fit a mini cell).
  compact?: boolean;
};

export function DeviationTimeline({ laps, reachedIndex, className = '', compact = false }: Props) {
  const max = gaugeMax(laps);
  const tau = stintThreshold(laps);
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

  const firstLap = laps[0]?.lap_number;
  const lastLap = laps[n - 1]?.lap_number;

  return (
    <div className={className} data-testid="deviation-timeline">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label="deviation timeline">
        {tau != null ? (
          <g data-testid="timeline-threshold-line">
            <line
              x1={PAD}
              y1={yOf(tau)}
              x2={W - PAD}
              y2={yOf(tau)}
              className="stroke-accent"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
            <text
              x={W - PAD}
              y={yOf(tau) - 2.5}
              textAnchor="end"
              className="fill-accent font-mono"
              fontSize={6}
            >
              τ {tau.toFixed(2)}
            </text>
          </g>
        ) : null}
        {segments.map((pts, i) => (
          <polyline key={i} points={pts} fill="none" className="stroke-text-secondary" strokeWidth={1.5} />
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
      {compact ? null : (
        <div className="mt-1 flex items-baseline justify-between gap-3 font-mono text-[9px] text-text-secondary">
          <span>{firstLap != null ? `Lap ${firstLap}` : ''}</span>
          <span data-testid="timeline-caption">
            {tau != null
              ? 'deviation per lap · dashed = alert threshold τ · gaps = unscored laps'
              : 'deviation per lap · gaps = unscored laps · τ unavailable (monitor not activated)'}
          </span>
          <span>{lastLap != null ? `Lap ${lastLap}` : ''}</span>
        </div>
      )}
    </div>
  );
}
