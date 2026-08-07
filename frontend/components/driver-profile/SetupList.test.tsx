import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { SetupResponse } from '@/lib/api/types';

import { SetupList } from './SetupList';

function makeSetup(overrides: Partial<SetupResponse> = {}): SetupResponse {
  return {
    driver: 'VER',
    circuit: 'Bahrain',
    parameters: [
      {
        name: 'front_wing_angle',
        kind: 'DEGREES',
        value: 4.401,
        value_range: null,
        category: null,
        confidence: 0.89,
        rating: 'HIGH',
        unit: 'degrees',
        fia_min: 0,
        fia_max: 12,
        constraint_flags: [],
        shap_top_features: [],
        shap_available: false,
      },
      {
        name: 'suspension_stiffness_front',
        kind: 'CATEGORY',
        value: 1,
        value_range: null,
        category: 'Medium',
        confidence: 0.7,
        rating: 'MEDIUM',
        unit: 'category',
        fia_min: null,
        fia_max: null,
        constraint_flags: [],
        shap_top_features: [],
        shap_available: false,
      },
      {
        name: 'brake_bias',
        kind: 'PERCENT',
        value: null,
        value_range: [54.0, 60.5],
        category: null,
        confidence: 0.4,
        rating: 'WIDE',
        unit: '% front',
        fia_min: 50,
        fia_max: 70,
        constraint_flags: ['CONSTRAINT_CLIPPED'],
        shap_top_features: [],
        shap_available: false,
      },
    ],
    consistency_warning: true,
    monaco_override: false,
    approximation_disclaimer: 'Setup values are approximated from telemetry signatures.',
    model_version: 'test-v1',
  } as SetupResponse;
}

describe('SetupList — the SetupPanel honesty rules, in compact rows', () => {
  it('renders every parameter with value + unit + rating chip', () => {
    render(<SetupList setup={makeSetup()} />);
    const row = screen.getByTestId('setup-param-front_wing_angle');
    expect(row).toHaveTextContent('Front wing angle');
    expect(row).toHaveTextContent('4.4 degrees');
    expect(row).toHaveTextContent('HIGH');
  });

  it('a CATEGORY parameter shows its label, never the raw ordinal', () => {
    render(<SetupList setup={makeSetup()} />);
    const row = screen.getByTestId('setup-param-suspension_stiffness_front');
    expect(row).toHaveTextContent('Medium');
    expect(within(row).queryByText('1')).toBeNull();
  });

  it('a WIDE parameter shows its range — a point value would overstate certainty', () => {
    render(<SetupList setup={makeSetup()} />);
    const row = screen.getByTestId('setup-param-brake_bias');
    expect(row).toHaveTextContent('54.0–60.5 % front');
    expect(row).toHaveTextContent('WIDE');
  });

  it('response warnings and constraint flags surface as chips — never dropped', () => {
    render(<SetupList setup={makeSetup()} />);
    const flags = screen.getByTestId('setup-flags');
    expect(flags).toHaveTextContent('CONSISTENCY_WARNING');
    expect(flags).toHaveTextContent('CONSTRAINT_CLIPPED');
  });

  it('always shows the approximation disclaimer', () => {
    render(<SetupList setup={makeSetup()} />);
    expect(screen.getByText(/approximated from telemetry signatures/)).toBeInTheDocument();
  });
});
