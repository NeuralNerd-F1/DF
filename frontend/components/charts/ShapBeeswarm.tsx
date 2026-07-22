'use client';

import { randomLcg } from 'd3';

import { LAYOUT_SEED } from '@/lib/design/animation';

export type BeeswarmLap = {
  lapNumber: number;
  shap: number;
  circuit: string;
  session?: string;
  confidence?: number;
};

// Colour-by-circuit — a stable palette indexed by the sorted circuit order (deterministic).
const PALETTE = ['#E8002D', '#3671C6', '#27F4D2', '#FF8000', '#229971', '#64C4FF', '#F5A623', '#9B59B6'];
function circuitColour(circuit: string, order: string[]): string {
  const i = order.indexOf(circuit);
  return PALETTE[(i < 0 ? 0 : i) % PALETTE.length];
}

// SHAP beeswarm (master §6/§10.3): one dot per lap, x = SHAP value, colour = circuit, hover = lap /
// session / confidence. The vertical jitter uses a FIXED-SEED LCG, so the same dataset renders
// byte-identically every time (reproducibility #3) — no Math.random() in layout.
export function ShapBeeswarm({
  laps,
  width = 320,
  height = 120,
  ariaLabel = 'SHAP beeswarm — one dot per lap',
}: {
  laps: BeeswarmLap[];
  width?: number;
  height?: number;
  ariaLabel?: string;
}) {
  const circuits = Array.from(new Set(laps.map((l) => l.circuit))).sort();
  const shaps = laps.map((l) => l.shap);
  const min = Math.min(-0.001, ...shaps);
  const max = Math.max(0.001, ...shaps);
  const pad = 8;
  const xOf = (s: number): number => pad + ((s - min) / (max - min)) * (width - 2 * pad);
  const rng = randomLcg(LAYOUT_SEED); // re-seeded every render → deterministic
  const cy0 = height / 2;
  const spread = height / 2 - pad;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
      data-testid="beeswarm"
    >
      <line x1={xOf(0)} y1={pad} x2={xOf(0)} y2={height - pad} className="stroke-border" strokeWidth={0.5} />
      {laps.map((l) => {
        const cx = Number(xOf(l.shap).toFixed(2));
        const cy = Number((cy0 + (rng() - 0.5) * 2 * spread).toFixed(2));
        const tip = `Lap ${l.lapNumber}${l.session ? ` · ${l.session}` : ''}${
          l.confidence != null ? ` · conf ${l.confidence.toFixed(2)}` : ''
        }`;
        return (
          <circle
            key={l.lapNumber}
            cx={cx}
            cy={cy}
            r={3}
            fill={circuitColour(l.circuit, circuits)}
            data-lap={l.lapNumber}
            data-circuit={l.circuit}
          >
            <title>{tip}</title>
          </circle>
        );
      })}
    </svg>
  );
}
