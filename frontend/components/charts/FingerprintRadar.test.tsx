import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { RADAR_DOMAINS } from '@/lib/api/types';

import { FingerprintRadar } from './FingerprintRadar';

const VALUES: Record<string, number> = {
  throttle: 0.5,
  braking: 0.3,
  speed: 0.7,
  gear: 0.4,
  engine: 0.6,
  cornering: 0.5,
};

describe('FingerprintRadar', () => {
  it('renders all six domain axes with their values + labels', () => {
    render(<FingerprintRadar values={VALUES} animate={false} />);
    for (const d of RADAR_DOMAINS) {
      const dot = document.querySelector(`circle[data-domain="${d}"]`);
      expect(dot, `missing vertex for ${d}`).not.toBeNull();
      expect(dot).toHaveAttribute('data-value', String(VALUES[d]));
    }
    expect(screen.getAllByText(/^(throttle|braking|speed|gear|engine|cornering)$/)).toHaveLength(6);
  });

  it('clamps out-of-range / NaN / missing domains to [0,1] — never a gap', () => {
    render(<FingerprintRadar values={{ throttle: 1.5, braking: -0.2, speed: NaN }} animate={false} />);
    expect(document.querySelector('circle[data-domain="throttle"]')).toHaveAttribute('data-value', '1');
    expect(document.querySelector('circle[data-domain="braking"]')).toHaveAttribute('data-value', '0');
    expect(document.querySelector('circle[data-domain="speed"]')).toHaveAttribute('data-value', '0');
    expect(document.querySelector('circle[data-domain="gear"]')).toHaveAttribute('data-value', '0');
  });

  it('dims the radar for the LOW_CONFIDENCE treatment', () => {
    render(<FingerprintRadar values={VALUES} animate={false} dimmed />);
    expect(screen.getByTestId('fingerprint-radar')).toHaveClass('opacity-40');
  });
});
