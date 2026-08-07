// Driver code → race number. Public F1 identity facts (like the team palette in teams.ts), not
// data the pipeline measures — kept here so the card can show the number fans recognise. A code
// with no entry falls back to rendering the code itself, never a fabricated number.
export const DRIVER_NUMBERS: Readonly<Record<string, number>> = {
  VER: 1, // raced #1 as champion across 2022–2025 (permanent 33 unused)
  HAM: 44,
  LEC: 16,
  ALO: 14,
  NOR: 4,
  RUS: 63,
  SAI: 55,
  GAS: 10,
  OCO: 31,
  ALB: 23,
  TSU: 22,
  HUL: 27,
  STR: 18,
  PER: 11,
  PIA: 81,
  BOT: 77,
  ZHO: 24,
  MAG: 20,
  RIC: 3,
  SAR: 2,
  LAW: 30,
  BEA: 87,
  ANT: 12,
  COL: 43,
  DOO: 7,
  BOR: 5,
  HAD: 6,
};

export function driverNumber(code: string): string {
  const n = DRIVER_NUMBERS[code];
  return n === undefined ? code : String(n);
}
