import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { DeviationGauge } from './DeviationGauge';

describe('DeviationGauge (div #72)', () => {
  it('shows the deviation score and draws a needle', () => {
    render(<DeviationGauge score={0.5} state="ALERT" max={1} />);
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
    render(<DeviationGauge score={null} state="BASELINE" max={1} />);
    expect(screen.getByTestId('deviation-gauge-value')).toHaveTextContent('—');
    expect(screen.queryByTestId('deviation-gauge-needle')).toBeNull();
    expect(screen.getByTestId('deviation-gauge')).toHaveAttribute('data-tone', 'idle');
  });

  it('labels the max as the client stint range with τ deferred (named seam)', () => {
    render(<DeviationGauge score={0.5} state="NOMINAL" max={1} />);
    expect(screen.getByTestId('gauge-threshold-note')).toHaveTextContent(/threshold not exposed/i);
  });
});
