import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { DeltaRadar } from './DeltaRadar';

describe('DeltaRadar (master §10.5, div #61)', () => {
  it('colours each axis dot by the sign of the delta (positive red, negative blue)', () => {
    const { container } = render(
      <DeltaRadar delta={{ throttle: 0.3, braking: -0.2 }} animate={false} />,
    );
    const throttle = container.querySelector('[data-domain="throttle"][data-sign]');
    const braking = container.querySelector('[data-domain="braking"][data-sign]');
    expect(throttle).toHaveAttribute('data-sign', 'positive');
    expect(throttle).toHaveClass('fill-delta-up');
    expect(braking).toHaveAttribute('data-sign', 'negative');
    expect(braking).toHaveClass('fill-delta-down');
  });

  it('renders a title with the signed value per axis', () => {
    const { container } = render(<DeltaRadar delta={{ throttle: 0.3 }} animate={false} />);
    const title = container.querySelector('circle[data-domain="throttle"] title');
    expect(title?.textContent).toContain('throttle: +0.30');
  });

  it('renders a flat radar (not a blank) when every delta is zero', () => {
    const zero = { throttle: 0, braking: 0, speed: 0, gear: 0, engine: 0, cornering: 0 };
    render(<DeltaRadar delta={zero} animate={false} />);
    expect(screen.getByTestId('delta-radar')).toBeInTheDocument();
    expect(screen.getByTestId('delta-polygon')).toBeInTheDocument();
  });
});
