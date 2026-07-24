import { describe, expect, it } from 'vitest';

import { ALERT_STATES, type ReplayLap } from '@/lib/api/types';

import {
  alertTone,
  deviationMax,
  firstAlertLap,
  hasAnyAlert,
  lapsInAlert,
  narrowSnapshot,
} from './replay';

const lap = (n: number, state: string, score: number | null): ReplayLap =>
  ({ lap_number: n, feature_vector: {}, alert_state: state, deviation_score: score, fingerprint_snapshot: null }) as ReplayLap;

const TRACE: ReplayLap[] = [
  lap(1, 'BASELINE', null),
  lap(2, 'NOMINAL', 0.2),
  lap(3, 'WATCH', 0.6),
  lap(4, 'ALERT', 0.9),
  lap(5, 'ALERT', 0.95),
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

describe('hasAnyAlert / firstAlertLap', () => {
  it('detects an ALERT anywhere in the replay', () => {
    expect(hasAnyAlert(TRACE)).toBe(true);
    expect(hasAnyAlert(TRACE.slice(0, 3))).toBe(false);
  });

  it('finds the first ALERT lap up to the reached frontier', () => {
    expect(firstAlertLap(TRACE, 4)?.lap_number).toBe(4);
    expect(firstAlertLap(TRACE, 2)).toBeNull(); // WATCH reached, no ALERT yet
  });
});

describe('lapsInAlert', () => {
  it('counts the contiguous ALERT run ending at the reached lap', () => {
    expect(lapsInAlert(TRACE, 4)).toBe(2); // laps 4 & 5
    expect(lapsInAlert(TRACE, 3)).toBe(1); // only lap 4 reached
  });

  it('returns 0 when the reached lap is not ALERT', () => {
    expect(lapsInAlert(TRACE, 2)).toBe(0);
    expect(lapsInAlert(TRACE, -1)).toBe(0);
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
