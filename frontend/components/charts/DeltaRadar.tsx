'use client';

import { easeCubicInOut, select } from 'd3';
import { useRef } from 'react';

import { RADAR_DOMAINS } from '@/lib/api/types';
import { DURATION } from '@/lib/design/animation';
import { useIsomorphicLayoutEffect } from '@/lib/useIsomorphicLayoutEffect';

// Signed delta radar (master §10.5): the per-domain difference between two drivers. Each axis plots
// |delta| as its radius; the vertex dot is RED when positive (driver-1 higher / toward) and BLUE
// when negative (driver-2 higher / away). Divergence #61: the backend `delta_shap` is a per-DOMAIN
// radar-value delta (6 domains), NOT a per-feature SHAP delta — labelled honestly. A zero delta
// collapses to the centre ring (a flat radar), never a blank.
type Props = {
  delta: Record<string, number>;
  size?: number;
  animate?: boolean;
  ariaLabel?: string;
  className?: string;
};

const RINGS = [0.25, 0.5, 0.75, 1] as const;

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

export function DeltaRadar({
  delta,
  size = 220,
  animate = true,
  ariaLabel = 'Delta radar (signed difference)',
  className = '',
}: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 30;
  const n = RADAR_DOMAINS.length;

  const angle = (i: number): number => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const point = (i: number, radial: number): [number, number] => {
    const a = angle(i);
    return [cx + r * radial * Math.cos(a), cy + r * radial * Math.sin(a)];
  };

  const magPts = RADAR_DOMAINS.map((d, i) => point(i, clamp01(Math.abs(delta[d] ?? 0))));
  const polygonPoints = magPts.map(([x, y]) => `${x},${y}`).join(' ');

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
        .duration(DURATION.deltaRadar)
        .ease(easeCubicInOut)
        .attr('points', polygonPoints);
    }
    dotRefs.current.forEach((dot, i) => {
      if (!dot) return;
      const [x, y] = magPts[i] ?? [cx, cy];
      select(dot)
        .attr('cx', cx)
        .attr('cy', cy)
        .transition()
        .duration(DURATION.deltaRadar)
        .ease(easeCubicInOut)
        .attr('cx', x)
        .attr('cy', y);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [polygonPoints, animate]);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      role="img"
      aria-label={ariaLabel}
      data-testid="delta-radar"
      className={className}
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
        const [lx, ly] = point(i, 1.18);
        return (
          <g key={d}>
            <line x1={cx} y1={cy} x2={ex} y2={ey} className="stroke-border" strokeWidth={0.5} />
            <text
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-text-secondary text-[9px] capitalize"
              data-domain={d}
            >
              {d}
            </text>
          </g>
        );
      })}
      <polygon
        ref={polyRef}
        points={polygonPoints}
        fill="none"
        className="stroke-text-secondary"
        strokeWidth={1}
        strokeDasharray="3 2"
        data-testid="delta-polygon"
      />
      {RADAR_DOMAINS.map((d, i) => {
        const value = delta[d] ?? 0;
        const positive = value >= 0;
        const [x, y] = magPts[i] ?? [cx, cy];
        return (
          <circle
            key={d}
            ref={(el) => {
              dotRefs.current[i] = el;
            }}
            cx={x}
            cy={y}
            r={3}
            className={positive ? 'fill-delta-up' : 'fill-delta-down'}
            data-domain={d}
            data-sign={positive ? 'positive' : 'negative'}
            data-value={value}
          >
            <title>{`${d}: ${positive ? '+' : ''}${value.toFixed(2)}`}</title>
          </circle>
        );
      })}
    </svg>
  );
}
