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
      category: null,
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
      category: null,
      confidence: 0.2,
      rating: 'WIDE',
      unit: 'deg',
      fia_min: 0,
      fia_max: 12,
      constraint_flags: ['WIDE_CONFIDENCE'],
      shap_top_features: [],
      shap_available: false,
    },
    {
      name: 'suspension_stiffness_front',
      kind: 'CATEGORY',
      value: 2, // ordinal — must NOT be shown as a needle/number
      value_range: null,
      category: 'Stiff', // the human label the panel must display
      confidence: 0.7,
      rating: 'MEDIUM',
      unit: null,
      fia_min: 0,
      fia_max: 2,
      constraint_flags: [],
      shap_top_features: [],
      shap_available: true,
    },
  ],
  consistency_warning: true,
  monaco_override: true,
  approximation_disclaimer: 'Setup targets are approximated from telemetry, not measured.',
  model_version: 'v',
} as unknown as SetupResponse;

describe('SetupPanel (master §10.3)', () => {
  it('renders a gauge per parameter, a range bar for WIDE, and the disclaimer', () => {
    render(<SetupPanel setup={setup} />);
    expect(screen.getAllByTestId('gauge')).toHaveLength(3);
    expect(screen.getByTestId('gauge-range')).toBeInTheDocument(); // the WIDE parameter
    expect(screen.getByText(/approximated from telemetry/)).toBeInTheDocument();
  });

  it('surfaces recommendation-level CONSISTENCY_WARNING + Monaco override, not just param flags (HIGH-1)', () => {
    render(<SetupPanel setup={setup} />);
    const flags = screen.getByTestId('setup-flags');
    expect(flags).toHaveTextContent('CONSISTENCY_WARNING');
    expect(flags).toHaveTextContent('CIRCUIT_OVERRIDE_MONACO');
    expect(flags).toHaveTextContent('WIDE_CONFIDENCE');
    expect(flags).toHaveTextContent('CONSTRAINT_CLIPPED');
  });

  it('shows a CATEGORY parameter as its label, not the raw ordinal or a needle (HIGH-2)', () => {
    render(<SetupPanel setup={setup} />);
    const gauges = screen.getAllByTestId('gauge');
    const suspension = gauges[2]; // suspension_stiffness_front
    expect(suspension).toHaveTextContent('Stiff');
    expect(suspension.querySelector('[data-testid="gauge-needle"]')).toBeNull();
  });
});
