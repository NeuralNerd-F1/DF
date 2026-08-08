import { describe, expect, it } from 'vitest';

import { featureShapDelta, pairAxisMaxima, scaleByMaxima } from './compare';

const R1 = { throttle: 3.2, braking: 0.4, speed: 0.8, gear: 1.0, engine: 0.2, cornering: 0.6 };
const R2 = { throttle: 1.6, braking: 0.9, speed: 0.9, gear: 0.2, engine: 1.5, cornering: 0.5 };

describe('pairAxisMaxima', () => {
  it('takes the per-domain max over both drivers', () => {
    const m = pairAxisMaxima(R1, R2);
    expect(m.throttle).toBeCloseTo(3.2);
    expect(m.braking).toBeCloseTo(0.9);
    expect(m.engine).toBeCloseTo(1.5);
  });

  it('treats missing / non-finite axes as 0, never NaN', () => {
    const m = pairAxisMaxima({ throttle: NaN } as Record<string, number>, {});
    expect(m.throttle).toBe(0);
    expect(m.braking).toBe(0);
  });
});

describe('scaleByMaxima', () => {
  it('scales both drivers onto ONE 0-1 scale per axis (the rim-clamp fix)', () => {
    const m = pairAxisMaxima(R1, R2);
    const s1 = scaleByMaxima(R1, m);
    const s2 = scaleByMaxima(R2, m);
    expect(s1.throttle).toBeCloseTo(1); // the pair max holds the rim...
    expect(s2.throttle).toBeCloseTo(1.6 / 3.2); // ...the other driver is proportional, not clamped
    expect(s2.engine).toBeCloseTo(1);
    expect(s1.engine).toBeCloseTo(0.2 / 1.5);
    for (const v of [...Object.values(s1), ...Object.values(s2)]) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('returns 0 on a zero-max axis (no divide-by-zero, no fabricated magnitude)', () => {
    const s = scaleByMaxima({ throttle: 0 }, { throttle: 0 });
    expect(s.throttle).toBe(0);
  });
});

describe('featureShapDelta', () => {
  it('is driver1 minus driver2 over the union of features', () => {
    const d = featureShapDelta({ F01: 0.3, F02: -0.1 }, { F01: 0.1, F03: 0.2 });
    expect(d.F01).toBeCloseTo(0.2);
    expect(d.F02).toBeCloseTo(-0.1); // absent on driver 2 → treated as 0
    expect(d.F03).toBeCloseTo(-0.2);
  });

  it('drops entries that are non-finite on BOTH sides', () => {
    const d = featureShapDelta({ F01: NaN }, { F01: NaN });
    expect('F01' in d).toBe(false);
  });
});
