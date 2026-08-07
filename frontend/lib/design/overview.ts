import type { DriverSummary } from '@/lib/api/types';

// Card order: most reliably identified driver first. identification_f1 is the champion's held-out
// per-driver score, so the grid reads as "who the model actually knows" — a real editorial
// ordering, not alphabet. Unscored drivers sink to the bottom; ties break on code so the order is
// deterministic for a given payload.
export function orderByIdentification(drivers: readonly DriverSummary[]): DriverSummary[] {
  const score = (d: DriverSummary): number => d.profile?.identification_f1 ?? -1;
  return [...drivers].sort((a, b) => score(b) - score(a) || a.code.localeCompare(b.code));
}

// The page subtitle's numbers, derived from the SAME payload the cards render — never a second
// source that could drift. Fields the payload can't answer are null and the subtitle omits them.
export function datasetSummary(drivers: readonly DriverSummary[]): {
  nDrivers: number;
  nLaps: number | null;
  seasonSpan: [number, number] | null;
} {
  let laps = 0;
  let sawLaps = false;
  const seasons: number[] = [];
  for (const d of drivers) {
    if (d.profile) {
      laps += d.profile.laps_analysed;
      sawLaps = true;
      seasons.push(...(d.profile.seasons ?? []));
    }
  }
  return {
    nDrivers: drivers.length,
    nLaps: sawLaps ? laps : null,
    seasonSpan: seasons.length > 0 ? [Math.min(...seasons), Math.max(...seasons)] : null,
  };
}
