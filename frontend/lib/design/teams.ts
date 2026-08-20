// Team → constructor-identity colours (pastel reference). Data layer ONLY: radar strokes, number
// glyphs, card rules, style-map points — never an area fill for layout. Keyed on the
// DriverSummary.team seam (div #47); an unknown or absent team falls back to the neutral
// `unavailable` token — never fabricated to a real team.
// Only the INK map survives: nothing in the presented screens paints a team-coloured area.
export const TEAM_COLOUR: Readonly<Record<string, string>> = {
  'Red Bull': '#3760AE',
  'Red Bull Racing': '#3760AE',
  Ferrari: '#C22F49',
  Mercedes: '#2A8879',
  McLaren: '#CE7A22',
  'Aston Martin': '#2C7860',
  Alpine: '#3781C2',
  Williams: '#37799B',
  RB: '#4E69BE',
  AlphaTauri: '#4E69BE',
  'Kick Sauber': '#3E9C56',
  'Alfa Romeo': '#B23A55',
  Haas: '#757A84',
};

// The neutral fallback equals the `unavailable` token (empty state, absent team).
export const NEUTRAL_TEAM_COLOUR = '#B4AEC0';

export function teamColour(team: string | null | undefined): string {
  if (!team) return NEUTRAL_TEAM_COLOUR;
  return TEAM_COLOUR[team] ?? NEUTRAL_TEAM_COLOUR;
}
