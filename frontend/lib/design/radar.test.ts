import { describe, expect, it } from 'vitest';

import { RADAR_DOMAINS } from '@/lib/api/types';

import { gridRelativeRadar, RADAR_MICRO_LABELS } from './radar';

const driver = (code: string, values: Record<string, number>) => ({
  code,
  radar_values: values,
});

describe('gridRelativeRadar — per-axis scaling to the grid maximum', () => {
  it('puts the grid-strongest driver at exactly 1.0 and scales the rest proportionally', () => {
    const out = gridRelativeRadar([
      driver('AAA', { throttle: 3.0, braking: 1.0 }),
      driver('BBB', { throttle: 1.5, braking: 0.5 }),
    ]);
    expect(out.AAA?.throttle).toBe(1.0);
    expect(out.BBB?.throttle).toBe(0.5);
    expect(out.AAA?.braking).toBe(1.0);
    expect(out.BBB?.braking).toBe(0.5);
  });

  it('never exceeds 1.0 — the clamp-against-the-rim lie is structurally impossible', () => {
    const out = gridRelativeRadar([
      driver('AAA', { throttle: 3.29, braking: 0.45, speed: 0.83, gear: 1.09, engine: 0.18, cornering: 0.63 }),
      driver('BBB', { throttle: 2.92, braking: 1.37, speed: 0.83, gear: 0.76, engine: 0.11, cornering: 0.44 }),
    ]);
    for (const code of ['AAA', 'BBB']) {
      for (const domain of RADAR_DOMAINS) {
        expect(out[code]?.[domain]).toBeGreaterThanOrEqual(0);
        expect(out[code]?.[domain]).toBeLessThanOrEqual(1);
      }
    }
  });

  it('a domain nobody scores on renders at the centre (0), not a mid-scale guess', () => {
    const out = gridRelativeRadar([driver('AAA', { throttle: 2.0 }), driver('BBB', {})]);
    expect(out.AAA?.braking).toBe(0);
    expect(out.BBB?.throttle).toBe(0);
  });

  it('non-finite values are treated as absent, not propagated into the scale', () => {
    const out = gridRelativeRadar([
      driver('AAA', { throttle: Number.NaN }),
      driver('BBB', { throttle: 2.0 }),
    ]);
    expect(out.AAA?.throttle).toBe(0);
    expect(out.BBB?.throttle).toBe(1);
  });

  it('has a micro label for every radar domain', () => {
    for (const domain of RADAR_DOMAINS) {
      expect(RADAR_MICRO_LABELS[domain]).toMatch(/^[A-Z]{3}$/);
    }
  });
});
