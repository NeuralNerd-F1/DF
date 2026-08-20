import { describe, expect, it } from 'vitest';

import type { SetupParameterSchema } from '@/lib/api/types';

import { SETUP_META, tendencyText } from './setup';

const param = (over: Partial<SetupParameterSchema>): SetupParameterSchema =>
  ({
    name: 'front_wing_angle',
    kind: 'DEGREES',
    value: 5.7,
    value_range: null,
    category: null,
    confidence: 0.8,
    rating: 'HIGH',
    unit: 'degrees',
    fia_min: 0,
    fia_max: 12,
    constraint_flags: [],
    shap_top_features: [],
    shap_available: true,
    grid_mean: 4.2,
    basis_features: ['F19', 'F14'],
    ...over,
  }) as unknown as SetupParameterSchema;

describe('tendencyText (system input #2 — honest directions only)', () => {
  it('states the direction vs the served grid average', () => {
    expect(tendencyText(param({}))).toBe(
      'more front downforce for sharper turn-in (vs grid average 4.2)',
    );
    expect(tendencyText(param({ value: 3.1 }))).toBe(
      'less front downforce for a calmer front end (vs grid average 4.2)',
    );
  });

  it('reads "in line" when the FORMATTED values are equal — no manufactured direction', () => {
    // 4.24 vs 4.16 both format to 4.2 at display precision → no direction claimed.
    expect(tendencyText(param({ value: 4.24, grid_mean: 4.16 }))).toBe(
      'in line with the grid average (4.2)',
    );
  });

  it('returns null without a served grid reference (no reference → no tendency)', () => {
    expect(tendencyText(param({ grid_mean: null }))).toBeNull();
  });

  it('returns null for a WIDE parameter (no point value to compare)', () => {
    expect(tendencyText(param({ value: null, value_range: [3, 5], rating: 'WIDE' }))).toBeNull();
  });

  it('uses integer formatting for INDEX kinds', () => {
    const t = tendencyText(
      param({ name: 'throttle_map_curve', kind: 'INDEX', value: 7, grid_mean: 5.4 }),
    );
    expect(t).toBe('more aggressive throttle response (vs grid average 5)');
  });

  it('covers all 10 parameters with more/less phrasing', () => {
    expect(Object.keys(SETUP_META)).toHaveLength(10);
    for (const meta of Object.values(SETUP_META)) {
      expect(meta.more).toBeTruthy();
      expect(meta.less).toBeTruthy();
      expect(meta.more).not.toBe(meta.less);
    }
  });
});
