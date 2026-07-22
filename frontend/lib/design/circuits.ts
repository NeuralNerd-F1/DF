// Circuit list for the Command Centre selector. Circuits are FIXED reference data (the 2022–24
// calendar), not a model output — the API exposes no /circuits endpoint (F16 spec §4.2 finding),
// so a static list is honest here (reference data, never fabricated inference). The held-out
// circuits (Belgium, Italy) are intentionally omitted — they are quarantined from training/eval
// (constitution / master), so no fingerprints exist for them to display.
export const CIRCUITS: readonly string[] = [
  'Bahrain',
  'Saudi Arabia',
  'Australia',
  'Miami',
  'Monaco',
  'Spain',
  'Canada',
  'Austria',
  'Britain',
  'Hungary',
  'Netherlands',
  'Singapore',
  'Japan',
  'United States',
  'Mexico',
  'Brazil',
  'Abu Dhabi',
];

export const DEFAULT_CIRCUIT: string = CIRCUITS[0] ?? 'Bahrain';
