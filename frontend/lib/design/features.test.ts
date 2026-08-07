import { describe, expect, it } from 'vitest';

import {
  FEATURE_LABELS,
  SIGNAL_HIGH_BAND,
  SIGNAL_LOW_BAND,
  signalDetail,
  signalText,
} from './features';

describe('signalText — direction words come ONLY from the grid percentile', () => {
  it('top band → the high phrase', () => {
    expect(signalText({ feature: 'F06', percentile: 1.0 })).toBe('Hard peak braking');
    expect(signalText({ feature: 'F06', percentile: SIGNAL_HIGH_BAND })).toBe('Hard peak braking');
  });

  it('bottom band → the low phrase', () => {
    expect(signalText({ feature: 'F06', percentile: 0.0 })).toBe('Gentle peak braking');
    expect(signalText({ feature: 'F06', percentile: SIGNAL_LOW_BAND })).toBe('Gentle peak braking');
  });

  it('mid band → the neutral phrase (a 0.5 rank justifies no direction)', () => {
    expect(signalText({ feature: 'F06', percentile: 0.5 })).toBe(
      'Distinctive peak brake pressure',
    );
  });

  it('missing percentile → neutral phrase, never a guessed direction', () => {
    expect(signalText({ feature: 'F06', percentile: null })).toBe(
      'Distinctive peak brake pressure',
    );
    expect(signalText({ feature: 'F06' })).toBe('Distinctive peak brake pressure');
  });

  it('unknown feature ID falls back to the raw ID — renders, never crashes', () => {
    expect(signalText({ feature: 'F99', percentile: 1.0 })).toBe('Distinctive F99');
  });

  it('every canonical feature has a label and directional phrases', () => {
    for (const fid of Object.keys(FEATURE_LABELS)) {
      // High and low must both produce feature-specific copy (not the neutral fallback).
      expect(signalText({ feature: fid, percentile: 1.0 })).not.toMatch(/^Distinctive/);
      expect(signalText({ feature: fid, percentile: 0.0 })).not.toMatch(/^Distinctive/);
    }
  });
});

describe('signalDetail — hover provenance carries the measured numbers', () => {
  it('names the feature, attribution and grid rank', () => {
    expect(signalDetail({ feature: 'F02', attribution: 1.1397, percentile: 1.0 })).toBe(
      'F02 throttle modulation — model attribution 1.14 · grid rank 100%',
    );
  });

  it('states "no grid rank" when the percentile is absent instead of inventing one', () => {
    expect(signalDetail({ feature: 'F02', attribution: 0.5, percentile: null })).toContain(
      'no grid rank',
    );
  });
});
