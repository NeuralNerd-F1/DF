'use client';

import { easeCubicInOut, select } from 'd3';
import { useRef } from 'react';

import { RADAR_DOMAINS, type RadarDomain } from '@/lib/api/types';
import { DURATION, STAGGER } from '@/lib/design/animation';
import { useIsomorphicLayoutEffect } from '@/lib/useIsomorphicLayoutEffect';

type Props = {
  values: Record<string, number>;
  size?: number;
  dimmed?: boolean; // reduced-opacity radar (the LOW_CONFIDENCE card still shows it — master §10.2)
  showLabels?: boolean;
  animate?: boolean; // false in tests → deterministic static render
  staggerMs?: number;
  durationMs?: number;
  onAxisHover?: (domain: RadarDomain | null) => void;
  ariaLabel?: string;
  className?: string;
};

const RINGS = [0.25, 0.5, 0.75, 1] as const;

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

// The 6-axis fingerprint radar (master §10.2/§10.3). React renders the deterministic structure;
// D3 drives the entrance bloom (grow from centre, per-axis stagger, easeCubicInOut). Values are
// clamped to [0,1]; a missing/NaN domain renders 0 (an axis at the centre), never a gap.
export function FingerprintRadar({
  values,
  size = 220,
  dimmed = false,
  showLabels = true,
  animate = true,
  staggerMs = STAGGER.screen2Axis,
  durationMs = DURATION.chart,
  onAxisHover,
  ariaLabel = 'Fingerprint radar',
  className = '',
}: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - (showLabels ? 30 : 8);
  const n = RADAR_DOMAINS.length;

  const angle = (i: number): number => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const point = (i: number, radial: number): [number, number] => {
    const a = angle(i);
    return [cx + r * radial * Math.cos(a), cy + r * radial * Math.sin(a)];
  };

  const dataPts = RADAR_DOMAINS.map((d, i) => point(i, clamp01(values[d] ?? 0)));
  const polygonPoints = dataPts.map(([x, y]) => `${x},${y}`).join(' ');

  const polyRef = useRef<SVGPolygonElement | null>(null);
  const dotRefs = useRef<Array<SVGCircleElement | null>>([]);

  useIsomorphicLayoutEffect(() => {
    if (!animate) return;
    const poly = polyRef.current;
    if (poly) {
      const collapsed = RADAR_DOMAINS.map(() => `${cx},${cy}`).join(' ');
      select(poly)
        .attr('points', collapsed)
        .transition()
        .duration(durationMs)
        .ease(easeCubicInOut)
        .attr('points', polygonPoints);
    }
    dotRefs.current.forEach((dot, i) => {
      if (!dot) return;
      const [x, y] = dataPts[i] ?? [cx, cy];
      select(dot)
        .attr('cx', cx)
        .attr('cy', cy)
        .attr('r', 0)
        .transition()
        .delay(i * staggerMs)
        .duration(durationMs)
        .ease(easeCubicInOut)
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 3);
    });
    // Re-run only when the shape changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [polygonPoints, animate]);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      role="img"
      aria-label={ariaLabel}
      data-testid="fingerprint-radar"
      className={`${dimmed ? 'opacity-40' : ''} ${className}`}
    >
      {RINGS.map((rr) => (
        <polygon
          key={rr}
          points={RADAR_DOMAINS.map((_, i) => point(i, rr).join(',')).join(' ')}
          fill="none"
          className="stroke-border"
          strokeWidth={0.5}
        />
      ))}
      {RADAR_DOMAINS.map((d, i) => {
        const [ex, ey] = point(i, 1);
        const [lx, ly] = point(i, 1.2);
        return (
          <g key={d}>
            <line x1={cx} y1={cy} x2={ex} y2={ey} className="stroke-border" strokeWidth={0.5} />
            {showLabels ? (
              <text
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-text-secondary text-[9px] capitalize"
                data-domain={d}
                onMouseEnter={onAxisHover ? () => onAxisHover(d) : undefined}
                onMouseLeave={onAxisHover ? () => onAxisHover(null) : undefined}
              >
                {d}
              </text>
            ) : null}
          </g>
        );
      })}
      <polygon
        ref={polyRef}
        points={polygonPoints}
        className="fill-accent/20 stroke-accent"
        strokeWidth={1.5}
        data-testid="radar-polygon"
      />
      {RADAR_DOMAINS.map((d, i) => {
        const [x, y] = dataPts[i] ?? [cx, cy];
        return (
          <circle
            key={d}
            ref={(el) => {
              dotRefs.current[i] = el;
            }}
            cx={x}
            cy={y}
            r={3}
            className="fill-accent"
            data-domain={d}
            data-value={clamp01(values[d] ?? 0)}
          />
        );
      })}
    </svg>
  );
}
