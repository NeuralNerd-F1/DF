'use client';

import Link from 'next/link';
import { useState } from 'react';

import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import { ConfidenceBadge } from '@/components/ds/ConfidenceBadge';
import type { ConfidenceState, DriverSummary } from '@/lib/api/types';
import { DURATION, STAGGER } from '@/lib/design/animation';
import { isLowConfidence } from '@/lib/design/confidence';
import { teamColour } from '@/lib/design/teams';

// A per-driver async scalar (compatibility / stability). A genuine fetch failure carries `error`
// (+ the `code`); honest absence is `value === null` with `error` false. `partial` marks 206 data.
// Never fabricated: absence renders "—", a failure renders a distinct, code-bearing marker.
export type Metric = {
  value: number | null;
  loading: boolean;
  error: boolean;
  code?: string;
  partial?: boolean;
};

// A failed metric is rendered DISTINCTLY from honest absence (failsafe HIGH-2): "⚠" + the code
// (in the monospace <dl>), never the same "—" a genuinely-absent metric shows. The failure is also
// logged at the fetch boundary (fetchApi), so it is observable, not silent.
function MetricValue({ m }: { m: Metric }) {
  if (m.loading) return <span className="text-text-secondary">…</span>;
  if (m.error) {
    return (
      <span className="text-accent" data-metric-error data-code={m.code} title={m.code ?? 'error'}>
        ⚠ {m.code ?? 'ERR'}
      </span>
    );
  }
  if (m.value === null) return <span>—</span>;
  const pct = Math.round(Math.max(0, Math.min(1, m.value)) * 100);
  return (
    <span title={m.partial ? 'Partial — fewer than 3 seasons of data' : undefined}>
      {pct}%
      {m.partial ? (
        <span className="text-text-secondary" data-testid="partial-marker">
          *
        </span>
      ) : null}
    </span>
  );
}

// Command Centre card (master §10.2): team-colour bar, animated fingerprint radar, circuit
// compatibility proxy, stability indicator, LOW_CONFIDENCE treatment, and a hover top-3 SHAP peek.
export function DriverCard({
  driver,
  compatibility,
  stability,
  confidenceState,
  animate = true,
}: {
  driver: DriverSummary;
  compatibility: Metric;
  stability: Metric;
  confidenceState?: ConfidenceState | null;
  animate?: boolean;
}) {
  const [hover, setHover] = useState(false);
  const low = isLowConfidence(confidenceState);
  const colour = teamColour(driver.team);
  const radar = driver.radar_values ?? {};
  const top3 = Object.entries(radar)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  const border = low ? 'border-dashed border-accent' : 'border-border hover:border-accent';

  return (
    <Link
      href={`/drivers/${driver.code}`}
      className="block focus:outline-none focus:ring-1 focus:ring-accent"
      data-testid={`driver-card-${driver.code}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div className={`overflow-hidden rounded-lg border bg-card transition-transform hover:scale-[1.02] ${border}`}>
        {/* team-colour bar — SVG fill (dynamic colour, no inline style) */}
        <svg viewBox="0 0 100 3" preserveAspectRatio="none" className="h-1 w-full" aria-hidden="true">
          <rect width={100} height={3} fill={colour} />
        </svg>
        <div className="p-3">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-mono text-lg text-text-primary">{driver.code}</span>
            <ConfidenceBadge state={confidenceState} />
          </div>
          <FingerprintRadar
            values={radar}
            size={140}
            showLabels={false}
            dimmed={low}
            animate={animate}
            staggerMs={STAGGER.screen1Axis}
            durationMs={DURATION.screen1Bloom}
            ariaLabel={`${driver.code} fingerprint`}
          />
          <dl className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px]">
            <div>
              <dt className="text-text-secondary">Compat</dt>
              <dd data-testid="compat" className="text-text-primary">
                <MetricValue m={compatibility} />
              </dd>
            </div>
            <div>
              <dt className="text-text-secondary">Stability</dt>
              <dd data-testid="stability" className="text-text-primary">
                <MetricValue m={stability} />
              </dd>
            </div>
          </dl>
          {hover ? (
            <div data-testid="hover-shap" className="mt-2 space-y-1">
              {top3.map(([label, value]) => {
                const w = Math.max(0, Math.min(1, value)) * 100;
                return (
                  <div key={label} className="flex items-center gap-1">
                    <span className="w-16 truncate font-mono text-[9px] capitalize text-text-secondary">
                      {label}
                    </span>
                    <svg viewBox="0 0 100 6" preserveAspectRatio="none" className="h-1.5 flex-1" aria-hidden="true">
                      <rect x={0} y={0} width={w} height={6} className="fill-accent">
                        {animate ? (
                          <animate attributeName="width" from="0" to={String(w)} dur="0.4s" fill="freeze" />
                        ) : null}
                      </rect>
                    </svg>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      </div>
    </Link>
  );
}
