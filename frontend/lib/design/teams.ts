// Team → constructor-identity colours (pastel reference). Data layer ONLY: radar strokes, number
// glyphs, card rules, style-map points — never an area fill for layout. Each team is an ink/fill
// pair; INK carries strokes and text, FILL carries chips and small areas. Keyed on the
// DriverSummary.team seam (div #47); an unknown or absent team falls back to the neutral
// `unavailable` token — never fabricated to a real team.
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

export const TEAM_FILL: Readonly<Record<string, string>> = {
  'Red Bull': '#CCD8F4',
  'Red Bull Racing': '#CCD8F4',
  Ferrari: '#F9CFD5',
  Mercedes: '#C6E8E0',
  McLaren: '#FCDEC2',
  'Aston Martin': '#CCE4D8',
  Alpine: '#D0E2F8',
  Williams: '#D2E5F0',
  RB: '#D3DAF7',
  AlphaTauri: '#D3DAF7',
  'Kick Sauber': '#CEE9D6',
  'Alfa Romeo': '#F5D2DB',
  Haas: '#E2E2E6',
};

// The neutral fallbacks equal the `unavailable` / `track` tokens (empty state, deferred seam).
export const NEUTRAL_TEAM_COLOUR = '#B4AEC0';
export const NEUTRAL_TEAM_FILL = '#E0DEE8';

export function teamColour(team: string | null | undefined): string {
  if (!team) return NEUTRAL_TEAM_COLOUR;
  return TEAM_COLOUR[team] ?? NEUTRAL_TEAM_COLOUR;
}

export function teamFill(team: string | null | undefined): string {
  if (!team) return NEUTRAL_TEAM_FILL;
  return TEAM_FILL[team] ?? NEUTRAL_TEAM_FILL;
}
