import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { StyleProfileCard } from './StyleProfileCard';

const RADAR = {
  throttle: 1.0,
  braking: 0.28,
  speed: 0.84,
  gear: 0.89,
  engine: 0.13,
  cornering: 0.5,
};

describe('StyleProfileCard', () => {
  it('renders a 0–100 score per domain from the grid-relative values', () => {
    render(<StyleProfileCard radar={RADAR} colour="#3671C6" code="VER" />);
    expect(screen.getByTestId('domain-score-throttle')).toHaveTextContent('100');
    expect(screen.getByTestId('domain-score-braking')).toHaveTextContent('28');
    expect(screen.getByTestId('domain-score-engine')).toHaveTextContent('13');
  });

  it('missing radar → dashes, not zero scores', () => {
    render(<StyleProfileCard radar={undefined} colour="#3671C6" code="VER" />);
    expect(screen.getByTestId('domain-score-throttle')).toHaveTextContent('—');
  });

  it('says what the numbers are — grid-max-scaled attribution', () => {
    render(<StyleProfileCard radar={RADAR} colour="#3671C6" code="VER" />);
    expect(screen.getByText(/scaled to the grid maximum/)).toBeInTheDocument();
  });
});
