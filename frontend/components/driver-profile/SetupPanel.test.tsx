import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { SetupResponse } from '@/lib/api/types';

import { SetupPanel } from './SetupPanel';

const setup = {
  driver: 'VER',
  circuit: 'Monaco',
  parameters: [
    {
      name: 'front_wing_angle',
      kind: 'DEGREES',
      value: 6,
      value_range: null,
      category: 'aero',
      confidence: 0.9,
      rating: 'HIGH',
      unit: 'deg',
      fia_min: 0,
      fia_max: 12,
      constraint_flags: ['CONSTRAINT_CLIPPED'],
      shap_top_features: [],
      shap_available: true,
    },
    {
      name: 'rear_wing_angle',
      kind: 'DEGREES',
      value: null,
      value_range: [2, 8],
      category: 'aero',
      confidence: 0.2,
      rating: 'WIDE',
      unit: 'deg',
      fia_min: 0,
      fia_max: 12,
      constraint_flags: ['WIDE_CONFIDENCE'],
      shap_top_features: [],
      shap_available: false,
    },
  ],
  consistency_warning: false,
  monaco_override: true,
  approximation_disclaimer: 'Setup targets are approximated from telemetry, not measured.',
  model_version: 'v',
} as unknown as SetupResponse;

describe('SetupPanel (master §10.3)', () => {
  it('renders a gauge per parameter, a range bar for WIDE, constraint flags, and the disclaimer', () => {
    render(<SetupPanel setup={setup} />);
    expect(screen.getAllByTestId('gauge')).toHaveLength(2);
    expect(screen.getByTestId('gauge-range')).toBeInTheDocument(); // the WIDE parameter
    expect(screen.getByTestId('setup-flags')).toHaveTextContent('WIDE_CONFIDENCE');
    expect(screen.getByTestId('setup-flags')).toHaveTextContent('CONSTRAINT_CLIPPED');
    expect(screen.getByText(/approximated from telemetry/)).toBeInTheDocument();
  });
});
