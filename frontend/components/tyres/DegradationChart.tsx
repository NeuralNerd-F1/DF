import type { TyreCompoundStats } from '@/lib/api/types';

const VIEW_W = 860;
const VIEW_H = 320;
const PAD_L = 52;
const PAD_R = 16;
const PAD_T = 16;
const PAD_B = 40;

// Fuel-corrected lap-time delta vs tyre age: the median line in the team colour, the IQR as a
// band, the estimated cliff (when one was detected) as a dashed vertical, and the age-1 warm-up
// point annotated for what it is. One honest axis — no invented "tyre wear %" overlay.
export function DegradationChart({
  stats,
  colour,
  label,
}: {
  stats: TyreCompoundStats;
  colour: string;
  label: string;
}) {
  const curve = stats.curve ?? [];
  if (curve.length < 3) {
    return (
      <p className="font-mono text-[11px] text-text-secondary" data-testid="curve-absent">
        Not enough laps at distinct tyre ages to draw a curve.
      </p>
    );
  }

  const ages = curve.map((c) => c.age);
  const [x0, x1] = [Math.min(...ages), Math.max(...ages)];
  // Domain from the post-warm-up curve only: the age-1 out-lap cost is several seconds and would
  // compress the tyre trend into a flat line. Points beyond the cap are clipped to the top edge
  // and labelled with their real value instead of silently rescaling the story away.
  const body = curve.filter((c) => c.age >= 2);
  const lows = body.map((c) => c.q25);
  const highs = body.map((c) => c.q75);
  const y0 = Math.min(...lows, 0);
  const y1 = Math.max(...highs, 0.1) * 1.12;
  const clampY = (d: number): number => Math.min(d, y1);
  const sx = (a: number): number =>
    PAD_L + ((a - x0) / (x1 - x0 || 1)) * (VIEW_W - PAD_L - PAD_R);
  const sy = (d: number): number =>
    VIEW_H - PAD_B - ((clampY(d) - y0) / (y1 - y0 || 1)) * (VIEW_H - PAD_T - PAD_B);

  const median = curve.map((c) => `${sx(c.age)},${sy(c.delta_s)}`).join(' ');
  const band = [
    ...curve.map((c) => `${sx(c.age)},${sy(c.q75)}`),
    ...[...curve].reverse().map((c) => `${sx(c.age)},${sy(c.q25)}`),
  ].join(' ');

  const yTicks = [-1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0].filter((t) => t >= y0 && t <= y1);
  const xStep = x1 - x0 > 24 ? 5 : 2;
  const xTicks = ages.filter((a) => a % xStep === 0);
  const warmUp = curve.find((c) => c.age === 1);
  const warmUpClipped = warmUp !== undefined && warmUp.delta_s > y1;

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="w-full"
      role="img"
      aria-label={`Fuel-corrected lap-time degradation for ${label}`}
      data-testid="degradation-chart"
    >
      {/* grid + axes */}
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={PAD_L} y1={sy(t)} x2={VIEW_W - PAD_R} y2={sy(t)} className="stroke-border" strokeWidth={0.5} />
          <text x={PAD_L - 8} y={sy(t) + 3} textAnchor="end" className="fill-text-secondary font-mono text-[10px]">
            +{t.toFixed(1)}s
          </text>
        </g>
      ))}
      {xTicks.map((a) => (
        <g key={`x${a}`}>
          <line x1={sx(a)} y1={PAD_T} x2={sx(a)} y2={VIEW_H - PAD_B} className="stroke-border" strokeWidth={0.5} />
          <text x={sx(a)} y={VIEW_H - PAD_B + 16} textAnchor="middle" className="fill-text-secondary font-mono text-[10px]">
            {a}
          </text>
        </g>
      ))}
      <text
        x={(PAD_L + VIEW_W - PAD_R) / 2}
        y={VIEW_H - 6}
        textAnchor="middle"
        className="fill-text-secondary font-mono text-[9px] uppercase tracking-[0.18em]"
      >
        Tyre age (laps)
      </text>

      {/* IQR band + median line */}
      <polygon points={band} fill={colour} fillOpacity={0.14} data-testid="iqr-band" />
      <polyline points={median} fill="none" stroke={colour} strokeWidth={2} data-testid="median-line" />
      {curve.map((c) => (
        <circle key={c.age} cx={sx(c.age)} cy={sy(c.delta_s)} r={2.5} fill={colour}>
          <title>{`age ${c.age}: +${c.delta_s.toFixed(2)}s median (IQR ${c.q25.toFixed(2)}…${c.q75.toFixed(2)}, n=${c.n})`}</title>
        </circle>
      ))}

      {/* warm-up annotation on the age-1 point; a clipped point states its real value */}
      {warmUp && (warmUp.delta_s > 0.5 || warmUpClipped) ? (
        <text
          x={sx(1) + 8}
          y={sy(warmUp.delta_s) + (warmUpClipped ? 10 : -6)}
          className="fill-text-secondary font-mono text-[9px]"
          data-testid="warmup-note"
        >
          out-lap / warm-up{warmUpClipped ? ` (+${warmUp.delta_s.toFixed(1)}s, above scale)` : ''}
        </text>
      ) : null}

      {/* estimated cliff */}
      {stats.cliff_age != null ? (
        <g data-testid="cliff-marker">
          <line
            x1={sx(stats.cliff_age)}
            y1={PAD_T}
            x2={sx(stats.cliff_age)}
            y2={VIEW_H - PAD_B}
            stroke={colour}
            strokeWidth={1}
            strokeDasharray="4 4"
          />
          <text
            x={sx(stats.cliff_age) + 6}
            y={PAD_T + 12}
            className="fill-text-primary font-mono text-[9px] uppercase tracking-[0.12em]"
          >
            Est. cliff · lap {stats.cliff_age}
          </text>
        </g>
      ) : null}
    </svg>
  );
}
