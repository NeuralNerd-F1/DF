'use client';

import { teamColour } from '@/lib/design/teams';
import type { StyleNode } from '@/lib/style-space/styleSpace';

// Legend for the style-space scatter. Without it the scene is just unlabelled coloured dots — this
// lists every plotted driver with its team colour so a point can be identified without hovering.
// A row hovers / selects the matching sphere (shared state with the scene), so the legend doubles as
// a driver picker. Sorted by team then code so team-mates group together.
export function StyleSpaceLegend({
  nodes,
  selected,
  onHover,
  onSelect,
}: {
  nodes: StyleNode[];
  selected: string | null;
  onHover: (code: string | null) => void;
  onSelect: (code: string) => void;
}) {
  const sorted = [...nodes].sort(
    (a, b) => (a.team ?? 'zzz').localeCompare(b.team ?? 'zzz') || a.code.localeCompare(b.code),
  );
  return (
    <div className="rounded-lg border border-border bg-card p-3" data-testid="style-space-legend">
      <h2 className="mb-2 font-mono text-[10px] uppercase tracking-wide text-text-secondary">
        Drivers ({nodes.length}) · colour = team
      </h2>
      <ul className="max-h-64 space-y-0.5 overflow-y-auto" onMouseLeave={() => onHover(null)}>
        {sorted.map((n) => (
          <li key={n.code}>
            <button
              type="button"
              onMouseEnter={() => onHover(n.code)}
              onFocus={() => onHover(n.code)}
              onClick={() => onSelect(n.code)}
              aria-pressed={selected === n.code}
              className={`flex w-full items-center gap-2 rounded px-1.5 py-1 text-left transition-colors hover:bg-border/40 ${
                selected === n.code ? 'bg-border/60' : ''
              }`}
              data-testid={`legend-row-${n.code}`}
            >
              {/* SVG fill (not a style prop) mirrors StyleSpace2D's swatch, satisfying the DOM-props rule. */}
              <svg viewBox="0 0 10 10" className="h-2.5 w-2.5 shrink-0" aria-hidden>
                <circle cx="5" cy="5" r="5" fill={teamColour(n.team)} />
              </svg>
              <span className="font-mono text-xs text-text-primary">{n.code}</span>
              <span className="ml-auto truncate pl-2 font-mono text-[10px] text-text-secondary">
                {n.team ?? 'no team'}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
