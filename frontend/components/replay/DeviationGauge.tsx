'use client';

import { ALERT_FILL_CLASS, ALERT_STROKE_CLASS, ALERT_TEXT_CLASS, alertTone } from '@/lib/replay/replay';

// The deviation-score gauge (master §10.7, div #72). Value = the current lap's deviation_score; the
// COLOUR comes from alert_state (NOMINAL→green / WATCH→amber / ALERT→red / else grey), NOT a numeric
// τ compare — τ (threshold) is dropped by the route. Gauge max = the client stint deviation range,
// labelled as such. deviation_score == null → "—" (honest, never 0). Named seam: ReplayLap.threshold.
type Props = { score: number | null; state: string; max: number; className?: string };

const CX = 50;
const CY = 50;
const R = 38;

function clampTo(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
function angleDeg(v: number, max: number): number {
  const f = max > 0 ? clampTo(v, 0, max) / max : 0;
  return 180 - f * 180; // 0 → 180° (left), max → 0° (right)
}
function pointAt(deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180;
  return [CX + R * Math.cos(rad), CY - R * Math.sin(rad)];
}
function arcPath(a0: number, a1: number): string {
  const [x0, y0] = pointAt(a0);
  const [x1, y1] = pointAt(a1);
  const large = Math.abs(a0 - a1) > 180 ? 1 : 0;
  return `M ${x0} ${y0} A ${R} ${R} 0 ${large} 0 ${x1} ${y1}`; // sweep 0 → the upper semicircle
}

export function DeviationGauge({ score, state, max, className = '' }: Props) {
  const tone = alertTone(state);
  const hasScore = typeof score === 'number' && Number.isFinite(score);
  const header = hasScore ? (score as number).toFixed(2) : '—';
  const needleDeg = hasScore ? angleDeg(score as number, max) : null;

  return (
    <div className={className} data-testid="deviation-gauge" data-tone={tone} data-state={state}>
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[10px] uppercase text-text-secondary">deviation</span>
        <span
          className={`font-mono text-sm ${ALERT_TEXT_CLASS[tone]}`}
          data-testid="deviation-gauge-value"
        >
          {header}
        </span>
      </div>
      <svg viewBox="0 0 100 56" className="w-full" aria-label={`deviation ${header}, ${state}`}>
        <path d={arcPath(180, 0)} fill="none" className="stroke-border" strokeWidth={3} />
        {hasScore && needleDeg != null ? (
          <>
            {/* value arc from 0 → score, coloured by alert_state */}
            <path
              d={arcPath(180, needleDeg)}
              fill="none"
              className={ALERT_STROKE_CLASS[tone]}
              strokeWidth={5}
              data-testid="deviation-gauge-arc"
            />
            <g data-testid="deviation-gauge-needle">
              {(() => {
                const [ex, ey] = pointAt(needleDeg);
                return (
                  <line x1={CX} y1={CY} x2={ex} y2={ey} className={ALERT_STROKE_CLASS[tone]} strokeWidth={1.5} />
                );
              })()}
              <circle cx={CX} cy={CY} r={2} className={ALERT_FILL_CLASS[tone]} />
            </g>
          </>
        ) : null}
      </svg>
      <p className="mt-1 font-mono text-[9px] text-text-secondary" data-testid="gauge-threshold-note">
        max = stint range (τ threshold not exposed)
      </p>
    </div>
  );
}
