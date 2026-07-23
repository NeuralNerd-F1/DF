'use client';

import { easeCubicInOut, select } from 'd3';
import { useRef } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { DURATION } from '@/lib/design/animation';
import { useIsomorphicLayoutEffect } from '@/lib/useIsomorphicLayoutEffect';

// SHAP force plot (master §10.4). One signed bar per feature: red extends right = TOWARD the
// prediction, blue extends left = AWAY. Divergence #56: /predict surfaces only the signed
// `shap_values` map + `shap_available`; the base value, human-readable names and toward/away
// direction are computed backend-side (ExplanationResult.force) but DROPPED at the API boundary —
// so there is NO base-value baseline line here, and the feature VALUE is read from the request
// vector the caller passes in. When SHAP is unavailable the prediction is preserved and this
// renders an honest EmptyState (never a blank/zeroed plot).
type Props = {
  shapValues: Record<string, number> | null | undefined;
  available: boolean;
  featureValues?: Record<string, number | null> | null; // the request vector (for hover value)
  maxBars?: number;
  animate?: boolean;
  className?: string;
};

function fmtVal(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}

export function ShapForcePlot({
  shapValues,
  available,
  featureValues,
  maxBars = 12,
  animate = true,
  className = '',
}: Props) {
  const entries = shapValues ? Object.entries(shapValues).filter(([, s]) => Number.isFinite(s)) : [];

  const rows = entries
    .slice()
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, maxBars);

  const maxAbs = rows.reduce((m, [, s]) => Math.max(m, Math.abs(s)), 0);
  const rectRefs = useRef<Array<SVGRectElement | null>>([]);

  useIsomorphicLayoutEffect(() => {
    if (!animate) return;
    rectRefs.current.forEach((rect, i) => {
      const row = rows[i];
      if (!rect || !row) return;
      const len = maxAbs > 0 ? (46 * Math.abs(row[1])) / maxAbs : 0;
      const x = row[1] >= 0 ? 50 : 50 - len;
      select(rect)
        .attr('x', 50)
        .attr('width', 0)
        .transition()
        .duration(DURATION.forceBar)
        .ease(easeCubicInOut)
        .attr('x', x)
        .attr('width', len);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shapValues, animate]);

  if (!available || rows.length === 0) {
    return (
      <EmptyState
        message="Explanation unavailable (SHAP_UNAVAILABLE). Prediction shown above."
        className={className}
      />
    );
  }

  return (
    <div data-testid="shap-force" role="img" aria-label="SHAP force plot" className={className}>
      {rows.map(([feature, shap], i) => {
        const len = maxAbs > 0 ? (46 * Math.abs(shap)) / maxAbs : 0;
        const toward = shap >= 0;
        const x = toward ? 50 : 50 - len;
        const valStr = fmtVal(featureValues?.[feature]);
        return (
          <div
            key={feature}
            data-testid="force-row"
            data-feature={feature}
            data-sign={toward ? 'toward' : 'away'}
            className="flex items-center gap-2 py-0.5"
          >
            <span className="w-24 shrink-0 truncate font-mono text-[10px] text-text-secondary">
              {feature}
            </span>
            <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="h-3 flex-1">
              <line x1={50} y1={0} x2={50} y2={10} className="stroke-border" strokeWidth={0.5} />
              <rect
                ref={(el) => {
                  rectRefs.current[i] = el;
                }}
                x={x}
                y={2}
                width={len}
                height={6}
                className={toward ? 'fill-delta-up' : 'fill-delta-down'}
              >
                <title>{`${feature}: value=${valStr}, SHAP=${shap.toFixed(3)} (${toward ? 'toward' : 'away'})`}</title>
              </rect>
            </svg>
            <span className="w-12 shrink-0 text-right font-mono text-[10px] text-text-primary">
              {shap >= 0 ? '+' : ''}
              {shap.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
