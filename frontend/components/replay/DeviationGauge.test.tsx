import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { DeviationGauge } from './DeviationGauge';

describe('DeviationGauge (div #72 closed — real τ)', () => {
  it('shows the deviation score and draws a needle', () => {
    render(<DeviationGauge score={0.5} state="ALERT" max={1} threshold={0.7} />);
    expect(screen.getByTestId('deviation-gauge-value')).toHaveTextContent('0.50');
    expect(screen.getByTestId('deviation-gauge-needle')).toBeInTheDocument();
    expect(screen.getByTestId('deviation-gauge-arc')).toBeInTheDocument();
  });

  it('colours by alert_state — NOMINAL/WATCH/ALERT → green/amber/red tones', () => {
    const { rerender } = render(<DeviationGauge score={0.2} state="NOMINAL" max={1} />);
    expect(screen.getByTestId('deviation-gauge')).toHaveAttribute('data-tone', 'nominal');
    rerender(<DeviationGauge score={0.5} state="WATCH" max={1} />);
    expect(screen.getByTestId('deviation-gauge')).toHaveAttribute('data-tone', 'watch');
    rerender(<DeviationGauge score={0.9} state="ALERT" max={1} />);
    expect(screen.getByTestId('deviation-gauge')).toHaveAttribute('data-tone', 'alert');
  });

  it('renders "—" and NO needle when the score is null (honest, never 0)', () => {
    render(<DeviationGauge score={null} state="BASELINE" max={1} threshold={0.7} />);
    expect(screen.getByTestId('deviation-gauge-value')).toHaveTextContent('—');
    expect(screen.queryByTestId('deviation-gauge-needle')).toBeNull();
    expect(screen.getByTestId('deviation-gauge')).toHaveAttribute('data-tone', 'idle');
  });

  it('marks the REAL τ on the arc and states its value', () => {
    render(<DeviationGauge score={0.5} state="NOMINAL" max={1} threshold={0.7} />);
    expect(screen.getByTestId('gauge-threshold-tick')).toBeInTheDocument();
    expect(screen.getByTestId('gauge-threshold-note')).toHaveTextContent('τ = 0.70');
  });

  it('omits the tick and says so when τ is unavailable (monitor never activated)', () => {
    render(<DeviationGauge score={null} state="INACTIVE" max={1} threshold={null} />);
    expect(screen.queryByTestId('gauge-threshold-tick')).toBeNull();
    expect(screen.getByTestId('gauge-threshold-note')).toHaveTextContent(/τ unavailable/i);
  });
});
