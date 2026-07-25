'use client';

// Season filter (master §10.6) — DEFERRED (div #64). `/stylespace` returns one position per driver
// and never populates `season`; the backend computes `season_coords_*` in StyleSpaceHandle but the
// route drops them. So the control is rendered as an explicit disabled state with a note — never a
// fabricated per-season motion. The named additive seam is StyleSpaceDriver.season_coords_*.
const SEASONS = ['2022', '2023', '2024'] as const;

export function SeasonFilter() {
  return (
    <div data-testid="season-filter">
      <div className="flex gap-1" role="group" aria-label="Season (unavailable)">
        {SEASONS.map((s) => (
          <button
            key={s}
            type="button"
            disabled
            data-testid="season-btn"
            className="cursor-not-allowed rounded border border-border px-2 py-1 font-mono text-xs text-text-secondary opacity-40"
          >
            {s}
          </button>
        ))}
      </div>
      <p className="mt-1 font-mono text-[10px] text-text-secondary">
        Per-season positions not exposed by the API — season filter deferred.
      </p>
    </div>
  );
}
