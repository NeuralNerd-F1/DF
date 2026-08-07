import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { TyreCompoundStats } from '@/lib/api/types';

import { DegradationChart } from './DegradationChart';

const base: TyreCompoundStats = {
  n_laps: 100,
  n_stints: 10,
  median_stint_len: 19,
  max_stint_len: 30,
  deg_slope_s_per_lap: 0.06,
  total_delta_s: 1.1,
  fuel_slope_s_per_lap: -0.06,
  cliff_age: null,
  curve: [
    { age: 1, delta_s: 3.4, q25: 2.8, q75: 4.0, n: 27 },
    { age: 2, delta_s: 0.0, q25: -0.2, q75: 0.2, n: 62 },
    { age: 10, delta_s: 0.4, q25: 0.1, q75: 0.8, n: 70 },
    { age: 20, delta_s: 0.9, q25: 0.4, q75: 1.4, n: 40 },
  ],
  f27_median: null,
  f28_median: null,
};

describe('DegradationChart', () => {
  it('draws the median line and IQR band', () => {
    render(<DegradationChart stats={base} colour="#3671C6" label="VER MEDIUM" />);
    expect(screen.getByTestId('median-line')).toBeInTheDocument();
    expect(screen.getByTestId('iqr-band')).toBeInTheDocument();
  });

  it('annotates the clipped warm-up point with its real value', () => {
    render(<DegradationChart stats={base} colour="#3671C6" label="VER MEDIUM" />);
    expect(screen.getByTestId('warmup-note')).toHaveTextContent('+3.4s, above scale');
  });

  it('shows a cliff marker only when one was detected', () => {
    render(<DegradationChart stats={base} colour="#3671C6" label="VER MEDIUM" />);
    expect(screen.queryByTestId('cliff-marker')).toBeNull();
    render(<DegradationChart stats={{ ...base, cliff_age: 17 }} colour="#3671C6" label="x" />);
    expect(screen.getByTestId('cliff-marker')).toHaveTextContent('Est. cliff · lap 17');
  });

  it('too few curve points → explicit absence, no chart', () => {
    render(
      <DegradationChart
        stats={{ ...base, curve: base.curve!.slice(0, 2) }}
        colour="#3671C6"
        label="x"
      />,
    );
    expect(screen.getByTestId('curve-absent')).toBeInTheDocument();
    expect(screen.queryByTestId('degradation-chart')).toBeNull();
  });
});
