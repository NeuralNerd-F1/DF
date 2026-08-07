import Link from 'next/link';

import type { DriverSummary, StyleSpaceResponse } from '@/lib/api/types';
import { teamColour } from '@/lib/design/teams';

const VIEW_W = 640;
const VIEW_H = 340;
const PAD = 40;

// Deterministic label layout: labels sit right of the dot, flip to the left near the right edge
// (no clipping), and a label landing within a line-height of an already-placed same-side label is
// nudged downward until clear — UMAP clusters routinely put two drivers a few px apart.
function labelledPoints(
  points: { code: string; coords_2d: number[] }[],
  sx: (x: number) => number,
  sy: (y: number) => number,
): { code: string; x: number; y: number; labelX: number; labelY: number; anchor: 'start' | 'end' }[] {
  const placed: { x: number; y: number; anchor: 'start' | 'end' }[] = [];
  return points.map((d) => {
    const x = sx(d.coords_2d[0]);
    const y = sy(d.coords_2d[1]);
    const anchor: 'start' | 'end' = x > VIEW_W - PAD - 45 ? 'end' : 'start';
    const labelX = anchor === 'start' ? x + 9 : x - 9;
    let labelY = y + 3;
    // Same-side collision → push down one line at a time (points are few; this terminates fast).
    while (
      placed.some(
        (p) => p.anchor === anchor && Math.abs(p.x - labelX) < 42 && Math.abs(p.y - labelY) < 11,
      )
    ) {
      labelY += 11;
    }
    placed.push({ x: labelX, y: labelY, anchor });
    return { code: d.code, x, y, labelX, labelY, anchor };
  });
}

// Screen 2 style map — the REAL F08 2D projection with the profiled driver highlighted. Dots are
// team-coloured and link to each driver's profile. The caption is deliberately honest about what
// UMAP axes are: proximity means similar driving style; the axes themselves carry no physical
// units, so (unlike the mock this screen is styled after) they are not given invented labels.
export function StyleMapCard({
  style,
  summaries,
  highlight,
}: {
  style: StyleSpaceResponse;
  summaries: DriverSummary[];
  highlight: string;
}) {
  const teamOf = new Map(summaries.map((d) => [d.code, d.team]));
  const points = (style.drivers ?? []).filter(
    (d) => Array.isArray(d.coords_2d) && d.coords_2d.length >= 2,
  );

  if (points.length === 0) {
    return (
      <section className="rounded-md border border-border bg-card p-4" data-testid="style-map-card">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
          Style map
        </h2>
        <p className="mt-2 font-mono text-[11px] text-text-secondary">No style space computed yet.</p>
      </section>
    );
  }

  const xs = points.map((d) => d.coords_2d[0]);
  const ys = points.map((d) => d.coords_2d[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x: number): number =>
    PAD + ((x - x0) / (x1 - x0 || 1)) * (VIEW_W - 2 * PAD);
  // SVG y grows downward; flip so "up" on screen matches increasing UMAP y.
  const sy = (y: number): number =>
    VIEW_H - PAD - ((y - y0) / (y1 - y0 || 1)) * (VIEW_H - 2 * PAD);

  return (
    <section className="rounded-md border border-border bg-card p-4" data-testid="style-map-card">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
            Style map — UMAP projection
          </h2>
          <p className="mt-1 font-mono text-[10px] leading-snug text-text-secondary">
            Drivers placed by behavioural similarity over all 28 features — closer means more alike.
            The axes are the projection&apos;s own coordinates and carry no physical units.
          </p>
        </div>
        <span className="rounded border border-border px-2 py-1 font-mono text-[9px] uppercase tracking-[0.16em] text-text-primary">
          {highlight} highlighted
        </span>
      </div>

      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        className="mt-3 w-full"
        role="img"
        aria-label={`Style map of ${points.length} drivers with ${highlight} highlighted`}
        data-testid="style-map-svg"
      >
        {/* quartile grid — orientation only, deliberately unlabelled */}
        {[0.25, 0.5, 0.75].map((t) => (
          <g key={t}>
            <line
              x1={PAD + t * (VIEW_W - 2 * PAD)}
              y1={PAD}
              x2={PAD + t * (VIEW_W - 2 * PAD)}
              y2={VIEW_H - PAD}
              className="stroke-border"
              strokeWidth={0.5}
            />
            <line
              x1={PAD}
              y1={PAD + t * (VIEW_H - 2 * PAD)}
              x2={VIEW_W - PAD}
              y2={PAD + t * (VIEW_H - 2 * PAD)}
              className="stroke-border"
              strokeWidth={0.5}
            />
          </g>
        ))}
        <rect
          x={PAD}
          y={PAD}
          width={VIEW_W - 2 * PAD}
          height={VIEW_H - 2 * PAD}
          fill="none"
          className="stroke-border"
          strokeWidth={0.75}
        />
        {labelledPoints(points, sx, sy).map((d) => {
          const colour = teamColour(teamOf.get(d.code));
          const isHighlight = d.code === highlight;
          return (
            <Link key={d.code} href={`/drivers/${d.code}`} aria-label={`Open ${d.code} profile`}>
              <g data-testid={`style-dot-${d.code}`} data-highlight={isHighlight || undefined}>
                {isHighlight ? (
                  <circle cx={d.x} cy={d.y} r={11} fill="none" stroke={colour} strokeWidth={1.5} />
                ) : null}
                <circle cx={d.x} cy={d.y} r={isHighlight ? 6 : 4.5} fill={colour} />
                <text
                  x={d.labelX}
                  y={d.labelY}
                  textAnchor={d.anchor}
                  className={`font-mono text-[10px] ${
                    isHighlight ? 'fill-text-primary' : 'fill-text-secondary'
                  }`}
                >
                  {d.code}
                </text>
              </g>
            </Link>
          );
        })}
      </svg>
    </section>
  );
}
