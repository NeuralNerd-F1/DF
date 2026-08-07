import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { SignalWeightsCard } from './SignalWeightsCard';

const SIGNED = { F16: 0.455, F20: 0.369, F02: 0.317, F03: 0.272, F10: 0.22, F13: 0.194, F01: 0.1 };

function renderCard(overrides: Partial<Parameters<typeof SignalWeightsCard>[0]> = {}) {
  return render(
    <SignalWeightsCard
      code="VER"
      shapSigned={SIGNED}
      nlDescription="This driver is characterised by committed throttle."
      signature={[{ feature: 'F02', attribution: 1.14, percentile: 1.0 }]}
      {...overrides}
    />,
  );
}

describe('SignalWeightsCard', () => {
  it('shows the top-6 by |signed mean|, largest first — F01 (7th) is cut', () => {
    renderCard();
    expect(screen.getByTestId('signed-bar-F16')).toBeInTheDocument();
    expect(screen.getByTestId('signed-bar-F13')).toBeInTheDocument();
    expect(screen.queryByTestId('signed-bar-F01')).toBeNull();
  });

  it('positive values read as confirms; a negative value reads as argues', () => {
    renderCard({ shapSigned: { F16: 0.4, F08: -0.2 } });
    expect(screen.getByTestId('signed-bar-F16')).toHaveAttribute('data-direction', 'confirms');
    expect(screen.getByTestId('signed-bar-F08')).toHaveAttribute('data-direction', 'argues');
  });

  it('the argues-against legend appears ONLY when a negative bar is on screen', () => {
    renderCard(); // all positive
    expect(screen.queryByText(/argues against/i)).toBeNull();
    renderCard({ shapSigned: { F16: 0.4, F08: -0.2 } });
    expect(screen.getByText(/argues against/i)).toBeInTheDocument();
  });

  it('empty signed attribution → explicit absence, never a sign derived from magnitudes', () => {
    renderCard({ shapSigned: {} });
    expect(screen.getByTestId('signed-absent')).toBeInTheDocument();
  });

  it('renders the model-written driver summary and the key signals', () => {
    renderCard();
    expect(screen.getByTestId('driver-summary')).toHaveTextContent('committed throttle');
    expect(screen.getByText('Strongly modulated throttle')).toBeInTheDocument();
  });

  it('names features in plain English, not raw IDs', () => {
    renderCard();
    expect(screen.getByText('Gear-hold time')).toBeInTheDocument();
  });
});
