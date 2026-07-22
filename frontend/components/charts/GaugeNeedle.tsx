'use client';

// Setup-parameter gauge (master §10.3). A HIGH/MEDIUM/LOW parameter shows a needle settling to the
// recommended value; a WIDE_CONFIDENCE parameter shows a RANGE arc instead of a point needle
// (F16 spec §4.4 / §6). A categorical (string) or absent value renders honestly ("—"), never faked.
type Props = {
  label: string;
  value: number | string | null;
  range?: [number, number] | null; // present iff WIDE
  min: number;
  max: number;
  unit?: string;
  rating: string; // HIGH | MEDIUM | LOW | WIDE
  className?: string;
};

const CX = 50;
const CY = 50;
const R = 38;

function clampTo(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
function angleDeg(v: number, min: number, max: number): number {
  const f = max > min ? (clampTo(v, min, max) - min) / (max - min) : 0.5;
  return 180 - f * 180; // min → 180° (left), max → 0° (right)
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
function fmt(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

export function GaugeNeedle({ label, value, range, min, max, unit, rating, className = '' }: Props) {
  const isWide = rating === 'WIDE' || range != null;
  const suffix = unit ? ` ${unit}` : '';
  const headerValue =
    isWide && range
      ? `${fmt(range[0])}–${fmt(range[1])}${suffix}`
      : typeof value === 'number'
        ? `${fmt(value)}${suffix}`
        : typeof value === 'string'
          ? value
          : '—';

  return (
    <div className={className} data-testid="gauge" data-rating={rating}>
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[10px] capitalize text-text-secondary">
          {label.replace(/_/g, ' ')}
        </span>
        <span className="font-mono text-[10px] text-text-primary" data-testid="gauge-value">
          {headerValue}
        </span>
      </div>
      <svg viewBox="0 0 100 56" className="w-full" aria-label={`${label} ${headerValue}`}>
        <path d={arcPath(180, 0)} fill="none" className="stroke-border" strokeWidth={3} />
        {isWide && range ? (
          <path
            d={arcPath(angleDeg(range[1], min, max), angleDeg(range[0], min, max))}
            fill="none"
            className="stroke-accent-deep"
            strokeWidth={5}
            data-testid="gauge-range"
          />
        ) : typeof value === 'number' ? (
          <g data-testid="gauge-needle">
            {(() => {
              const [ex, ey] = pointAt(angleDeg(value, min, max));
              return <line x1={CX} y1={CY} x2={ex} y2={ey} className="stroke-accent" strokeWidth={1.5} />;
            })()}
            <circle cx={CX} cy={CY} r={2} className="fill-accent" />
          </g>
        ) : null}
      </svg>
    </div>
  );
}
