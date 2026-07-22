'use client';

import { useState } from 'react';

import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import type { RadarDomain } from '@/lib/api/types';
import { STAGGER } from '@/lib/design/animation';

// Natural-language meaning per domain (master §10.3 tooltip: domain, score, NL meaning, field-mean).
const MEANING: Record<RadarDomain, string> = {
  throttle: 'throttle application aggression',
  braking: 'braking intensity & timing',
  speed: 'top-end & mid-corner speed',
  gear: 'gear-hold & shift style',
  engine: 'power-unit deployment',
  cornering: 'cornering commitment',
};

// The large Screen-2 radar with a hover tooltip. Field mean is computed client-side over the 8
// drivers (F16 spec div #51) and passed in; the tooltip shows the driver's score vs that mean.
export function ProfileRadar({
  values,
  fieldMean,
  size = 300,
  animate = true,
}: {
  values: Record<string, number>;
  fieldMean?: Record<string, number>;
  size?: number;
  animate?: boolean;
}) {
  const [hover, setHover] = useState<RadarDomain | null>(null);
  const score = hover != null ? (values[hover] ?? 0) : 0;
  const mean = hover != null && fieldMean ? fieldMean[hover] : undefined;

  return (
    <div className="relative" data-testid="profile-radar">
      <FingerprintRadar
        values={values}
        size={size}
        showLabels
        animate={animate}
        staggerMs={STAGGER.screen2Axis}
        onAxisHover={setHover}
        ariaLabel="Driver fingerprint radar"
      />
      {hover != null ? (
        <div
          role="tooltip"
          data-testid="radar-tooltip"
          className="mt-2 rounded border border-border bg-card p-2 font-mono text-[10px] text-text-primary"
        >
          <span className="capitalize text-accent">{hover}</span> — {score.toFixed(2)}
          {mean != null ? (
            <span className="text-text-secondary">
              {' '}
              · field mean {mean.toFixed(2)} ({score >= mean ? '+' : ''}
              {(score - mean).toFixed(2)})
            </span>
          ) : null}
          <div className="text-text-secondary">{MEANING[hover]}</div>
        </div>
      ) : null}
    </div>
  );
}
