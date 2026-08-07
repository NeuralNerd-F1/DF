import { describe, expect, it } from 'vitest';

import type { DriverSummary } from '@/lib/api/types';

import { datasetSummary, orderByIdentification } from './overview';

const d = (
  code: string,
  f1: number | null,
  extra?: Partial<NonNullable<DriverSummary['profile']>>,
): DriverSummary =>
  ({
    id: code,
    code,
    team: null,
    radar_values: {},
    signature: [],
    profile: { laps_analysed: 100, seasons: [2024], identification_f1: f1, ...extra },
  }) as DriverSummary;

describe('orderByIdentification', () => {
  it('sorts by held-out F1 descending — the grid reads "who the model knows"', () => {
    const out = orderByIdentification([d('STR', 0.11), d('LEC', 0.68), d('VER', 0.53)]);
    expect(out.map((x) => x.code)).toEqual(['LEC', 'VER', 'STR']);
  });

  it('unscored drivers sink to the bottom; ties break on code deterministically', () => {
    const noProfile = { id: 'XXX', code: 'XXX', team: null, radar_values: {} } as DriverSummary;
    const out = orderByIdentification([noProfile, d('BBB', 0.5), d('AAA', 0.5), d('CCC', null)]);
    expect(out.map((x) => x.code)).toEqual(['AAA', 'BBB', 'CCC', 'XXX']);
  });

  it('does not mutate the input array', () => {
    const input = [d('AAA', 0.1), d('BBB', 0.9)];
    orderByIdentification(input);
    expect(input.map((x) => x.code)).toEqual(['AAA', 'BBB']);
  });
});

describe('datasetSummary', () => {
  it('sums laps and spans seasons from the same payload the cards render', () => {
    const out = datasetSummary([
      d('AAA', 0.5, { laps_analysed: 4000, seasons: [2022, 2023] }),
      d('BBB', 0.4, { laps_analysed: 3000, seasons: [2024, 2025] }),
    ]);
    expect(out).toEqual({ nDrivers: 2, nLaps: 7000, seasonSpan: [2022, 2025] });
  });

  it('reports null, not zero, when no driver carries a profile', () => {
    const bare = { id: 'AAA', code: 'AAA', team: null, radar_values: {} } as DriverSummary;
    expect(datasetSummary([bare])).toEqual({ nDrivers: 1, nLaps: null, seasonSpan: null });
  });
});
