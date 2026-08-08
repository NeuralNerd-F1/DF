import { describe, expect, it } from 'vitest';

import { ALERT_STATES, type ReplayLap } from '@/lib/api/types';

import {
  alertTone,
  deviationMax,
  driftTrail,
  gaugeMax,
  narrowSnapshot,
  stintThreshold,
  THRESHOLD_HEADROOM,
} from './replay';

const lap = (
  n: number,
  state: string,
  score: number | null,
  extra: Partial<ReplayLap> = {},
): ReplayLap =>
  ({
    lap_number: n,
    feature_vector: {},
    alert_state: state,
    deviation_score: score,
    fingerprint_snapshot: null,
    threshold: null,
    alert: null,
    ...extra,
  }) as ReplayLap;

const TRACE: ReplayLap[] = [
  lap(1, 'BASELINE', null),
  lap(2, 'NOMINAL', 0.2, { threshold: 0.7 }),
  lap(3, 'WATCH', 0.6, { threshold: 0.7 }),
  lap(4, 'ALERT', 0.9, { threshold: 0.7 }),
  lap(5, 'ALERT', 0.95, { threshold: 0.7 }),
];

describe('alertTone', () => {
  it('maps the three live states to green/amber/red tones', () => {
    expect(alertTone('NOMINAL')).toBe('nominal');
    expect(alertTone('WATCH')).toBe('watch');
    expect(alertTone('ALERT')).toBe('alert');
  });

  it('maps every other monitor state (incl. HALTED) to idle grey', () => {
    for (const s of ['INACTIVE', 'BASELINE', 'SKIPPED', 'PENDING', 'HALTED']) {
      expect(alertTone(s)).toBe('idle');
    }
    expect(alertTone(null)).toBe('idle');
    expect(alertTone(undefined)).toBe('idle');
  });

  it('returns a defined tone for all 8 wire states', () => {
    for (const s of ALERT_STATES) {
      expect(['nominal', 'watch', 'alert', 'idle']).toContain(alertTone(s));
    }
  });
});

describe('stintThreshold', () => {
  it('finds the stint τ past null BASELINE laps (constant per stint)', () => {
    expect(stintThreshold(TRACE)).toBeCloseTo(0.7);
  });

  it('is null when the monitor never activated (no lap carries τ)', () => {
    expect(stintThreshold([lap(1, 'BASELINE', null)])).toBeNull();
  });
});

describe('deviationMax', () => {
  it('is the max finite score across the replay', () => {
    expect(deviationMax(TRACE)).toBeCloseTo(0.95);
  });

  it('falls back to 1 when every score is null (no divide-by-zero)', () => {
    expect(deviationMax([lap(1, 'BASELINE', null)])).toBe(1);
  });
});

describe('gaugeMax', () => {
  it('anchors to τ with headroom when deviations stay under it', () => {
    const calm = [lap(1, 'NOMINAL', 0.2, { threshold: 0.7 })];
    expect(gaugeMax(calm)).toBeCloseTo(0.7 * THRESHOLD_HEADROOM);
  });

  it('lets the observed stint range win when deviations overshoot τ', () => {
    expect(gaugeMax(TRACE)).toBeCloseTo(0.95); // 0.95 > 0.7 × 1.25 = 0.875
  });

  it('falls back to the observed range when τ is unavailable', () => {
    expect(gaugeMax([lap(1, 'NOMINAL', 0.4)])).toBeCloseTo(0.4);
  });
});

describe('driftTrail', () => {
  const withCoords = (n: number, coords: unknown): ReplayLap =>
    lap(n, 'NOMINAL', 0.1, { fingerprint_snapshot: { coords_2d: coords } } as Partial<ReplayLap>);

  it('collects the reached laps’ 2D coords in order, tagged with their index', () => {
    const laps = [withCoords(1, [0, 0]), withCoords(2, [1, 2]), withCoords(3, [2, 4])];
    expect(driftTrail(laps, 1)).toEqual([
      { x: 0, y: 0, index: 0 },
      { x: 1, y: 2, index: 1 },
    ]);
  });

  it('skips laps without a snapshot or with non-finite coords — never fabricates a point', () => {
    const laps = [
      withCoords(1, [0, 1]),
      lap(2, 'NOMINAL', 0.2),
      withCoords(3, [NaN, 2]),
      withCoords(4, ['a', 'b']),
      withCoords(5, [3, 4]),
    ];
    expect(driftTrail(laps, 4)).toEqual([
      { x: 0, y: 1, index: 0 },
      { x: 3, y: 4, index: 4 },
    ]);
  });

  it('is empty for an out-of-range or negative frontier', () => {
    expect(driftTrail([withCoords(1, [1, 1])], -1)).toEqual([]);
  });
});

describe('narrowSnapshot', () => {
  it('narrows the open dict to coords + the real confidence_state', () => {
    const s = narrowSnapshot({ coords_5d: [1, 2, 3, 4, 5], coords_2d: [1, 2], confidence_state: 'HIGH' });
    expect(s?.confidence_state).toBe('HIGH');
    expect(s?.coords_5d).toHaveLength(5);
  });

  it('returns null for a null/absent snapshot', () => {
    expect(narrowSnapshot(null)).toBeNull();
    expect(narrowSnapshot(undefined)).toBeNull();
  });

  it('tolerates a snapshot missing confidence_state (coords only)', () => {
    const s = narrowSnapshot({ coords_5d: [0, 0, 0, 0, 0] });
    expect(s?.confidence_state).toBeNull();
  });
});
