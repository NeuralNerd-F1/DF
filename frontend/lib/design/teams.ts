// Team → colour-bar hex. The master (§10.2) names a "team colour bar" from drivers.team but
// enumerates NO palette (F16 spec §4.1 gap). Keyed on the DriverSummary.team seam (div #47); an
// unknown or absent team falls back to the neutral border token — never fabricated to a real team.
export const TEAM_COLOUR: Readonly<Record<string, string>> = {
  'Red Bull': '#3671C6',
  'Red Bull Racing': '#3671C6',
  Ferrari: '#E8002D',
  Mercedes: '#27F4D2',
  McLaren: '#FF8000',
  'Aston Martin': '#229971',
  Alpine: '#0093CC',
  Williams: '#64C4FF',
  RB: '#6692FF',
  AlphaTauri: '#6692FF',
  'Kick Sauber': '#52E252',
  'Alfa Romeo': '#C92D4B',
  Haas: '#B6BABD',
};

// The neutral fallback equals the §10.1 border token.
export const NEUTRAL_TEAM_COLOUR = '#2B2B2B';

export function teamColour(team: string | null | undefined): string {
  if (!team) return NEUTRAL_TEAM_COLOUR;
  return TEAM_COLOUR[team] ?? NEUTRAL_TEAM_COLOUR;
}
