import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { GaugeNeedle } from './GaugeNeedle';

describe('GaugeNeedle (master §10.3)', () => {
  it('shows a needle + value for a point (HIGH) parameter', () => {
    render(<GaugeNeedle label="front_wing_angle" value={6} min={0} max={12} unit="deg" rating="HIGH" />);
    expect(screen.getByTestId('gauge-needle')).toBeInTheDocument();
    expect(screen.queryByTestId('gauge-range')).toBeNull();
    expect(screen.getByTestId('gauge-value')).toHaveTextContent('6 deg');
  });

  it('shows a RANGE bar (no needle) for a WIDE_CONFIDENCE parameter', () => {
    render(
      <GaugeNeedle label="front_wing_angle" value={null} range={[2, 8]} min={0} max={12} unit="deg" rating="WIDE" />,
    );
    expect(screen.getByTestId('gauge-range')).toBeInTheDocument();
    expect(screen.queryByTestId('gauge-needle')).toBeNull();
    expect(screen.getByTestId('gauge-value')).toHaveTextContent('2 to 8 deg');
  });

  it('renders "—" for an absent value and a string for a categorical value (never faked)', () => {
    const { rerender } = render(<GaugeNeedle label="x" value={null} min={0} max={1} rating="LOW" />);
    expect(screen.getByTestId('gauge-value')).toHaveTextContent('—');
    rerender(<GaugeNeedle label="suspension" value="stiff" min={0} max={1} rating="MEDIUM" />);
    expect(screen.getByTestId('gauge-value')).toHaveTextContent('stiff');
    expect(screen.queryByTestId('gauge-needle')).toBeNull();
  });
});
