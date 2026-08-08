import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ReplayAlert, ReplayLap } from '@/lib/api/types';

import { AlertBanner } from './AlertBanner';

const ALERT: ReplayAlert = {
  alert_lap: 18,
  first_crossing_lap: 16,
  laps_since_threshold_crossed: 3,
  deviation_score: 1.4,
  threshold: 0.9,
  alert_confidence: 'HIGH',
  feature_attribution: { F27: 0.61, F03: 0.24 },
  top_features: ['F27', 'F03'],
  optimal_window_remaining_laps: 4,
};

const lap = (state: string, alert: ReplayAlert | null): ReplayLap =>
  ({
    lap_number: 20,
    feature_vector: {},
    alert_state: state,
    deviation_score: 1.4,
    fingerprint_snapshot: null,
    threshold: 0.9,
    alert,
  }) as ReplayLap;

describe('AlertBanner (div #74 closed — real payload)', () => {
  it('renders the served alert payload: lap, confidence, laps since crossing, pit window', () => {
    render(<AlertBanner lap={lap('ALERT', ALERT)} />);
    expect(screen.getByTestId('alert-banner')).toBeInTheDocument();
    expect(screen.getByTestId('alert-lap')).toHaveTextContent('Lap 18');
    expect(screen.getByTestId('alert-confidence')).toHaveTextContent('HIGH confidence');
    expect(screen.getByTestId('alert-laps-since')).toHaveTextContent('3 laps since threshold crossed');
    expect(screen.getByTestId('alert-pit-window')).toHaveTextContent('pit window: 4 laps remaining');
  });

  it('renders the feature drivers with their attribution weights', () => {
    render(<AlertBanner lap={lap('ALERT', ALERT)} />);
    expect(screen.getByTestId('alert-feature-drivers')).toBeInTheDocument();
    expect(screen.getByTestId('alert-driver-F27')).toHaveTextContent('0.61');
    expect(screen.getByTestId('alert-driver-F03')).toHaveTextContent('0.24');
  });

  it('does NOT render before the alert fires (alert=null — the monitor knew nothing yet)', () => {
    render(<AlertBanner lap={lap('WATCH', null)} />);
    expect(screen.queryByTestId('alert-banner')).toBeNull();
  });

  it('does NOT render without a lap at all', () => {
    render(<AlertBanner lap={null} />);
    expect(screen.queryByTestId('alert-banner')).toBeNull();
  });

  it('states honestly when the pit window is unavailable', () => {
    render(
      <AlertBanner lap={lap('ALERT', { ...ALERT, optimal_window_remaining_laps: null })} />,
    );
    expect(screen.getByTestId('alert-pit-window')).toHaveTextContent('pit window unavailable');
  });

  it('marks a recovered stint (payload present, state back under τ) as past-alert, not active', () => {
    render(<AlertBanner lap={lap('NOMINAL', ALERT)} />);
    const banner = screen.getByTestId('alert-banner');
    expect(banner).not.toHaveAttribute('data-active');
    expect(screen.getByTestId('alert-recovered')).toBeInTheDocument();
  });

  it('uses the singular for one lap since crossing', () => {
    render(<AlertBanner lap={lap('ALERT', { ...ALERT, laps_since_threshold_crossed: 1 })} />);
    expect(screen.getByTestId('alert-laps-since')).toHaveTextContent('1 lap since threshold crossed');
  });
});
