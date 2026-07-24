'use client';

import { teamColour } from '@/lib/design/teams';
import type { StyleNode, TeammateLink } from '@/lib/style-space/styleSpace';

// D3/SVG 2D scatter — the silent WebGL-unavailable fallback (master §10.6, div #68) and the
// jsdom-testable path. One <circle> per driver at its coords_2d, filled with teamColour (SVG `fill`
// attribute, not a forbidden inline style). Teammate <line>s thickness ∝ intra-team similarity.
// Deterministic (no animation, no random) — hover/click are delegated to the parent.
type Props = {
  nodes: StyleNode[];
  teammates: TeammateLink[];
  showTeammates: boolean;
  selected: string | null;
  onSelect: (code: string) => void;
  onHover: (code: string | null) => void;
  size?: number;
};

const PAD = 28;

function extent(vals: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of vals) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo - 1, hi + 1];
  return [lo, hi];
}

export function StyleSpace2D({
  nodes,
  teammates,
  showTeammates,
  selected,
  onSelect,
  onHover,
  size = 480,
}: Props) {
  if (nodes.length === 0) return null;

  const [x0, x1] = extent(nodes.map((n) => n.coords2d[0]));
  const [y0, y1] = extent(nodes.map((n) => n.coords2d[1]));
  const sx = (v: number): number => PAD + ((v - x0) / (x1 - x0)) * (size - 2 * PAD);
  const sy = (v: number): number => PAD + ((v - y0) / (y1 - y0)) * (size - 2 * PAD);
  const posOf = new Map(nodes.map((n) => [n.code, { x: sx(n.coords2d[0]), y: sy(n.coords2d[1]) }]));
  const maxSim = Math.max(...teammates.map((t) => t.similarity), 1e-9);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      role="img"
      aria-label="Style space (2D fallback)"
      data-testid="style-space-2d"
      className="max-w-full"
    >
      {showTeammates
        ? teammates.map((t) => {
            const a = posOf.get(t.a);
            const b = posOf.get(t.b);
            if (!a || !b) return null;
            const w = 0.5 + 3 * (t.similarity / maxSim);
            return (
              <line
                key={`${t.a}-${t.b}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                className="stroke-border"
                strokeWidth={w}
                data-testid="teammate-line"
              />
            );
          })
        : null}
      {nodes.map((n) => {
        const p = posOf.get(n.code);
        if (!p) return null;
        const isSel = n.code === selected;
        return (
          <circle
            key={n.code}
            cx={p.x}
            cy={p.y}
            r={isSel ? 9 : 6}
            fill={teamColour(n.team)}
            stroke={isSel ? '#F0F0F0' : 'none'}
            strokeWidth={isSel ? 2 : 0}
            className="cursor-pointer"
            data-testid="style-node"
            data-driver={n.code}
            data-team={n.team ?? ''}
            onMouseEnter={() => onHover(n.code)}
            onMouseLeave={() => onHover(null)}
            onClick={() => onSelect(n.code)}
          >
            <title>{n.code}</title>
          </circle>
        );
      })}
    </svg>
  );
}
