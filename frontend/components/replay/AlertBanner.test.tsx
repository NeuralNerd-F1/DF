import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ReplayLap } from '@/lib/api/types';

import { AlertBanner } from './AlertBanner';

const lap = (n: number, state: string): ReplayLap =>
  ({ lap_number: n, feature_vector: {}, alert_state: state, deviation_score: 0.5, fingerprint_snapshot: null }) as ReplayLap;

const LAPS: ReplayLap[] = [lap(1, 'NOMINAL'), lap(2, 'WATCH'), lap(3, 'ALERT'), lap(4, 'ALERT')];

describe('AlertBanner (div #74)', () => {
  it('fires on a reached ALERT lap with the lap number + client-derived laps-in-alert', () => {
    render(<AlertBanner laps={LAPS} reachedIndex={3} />);
    expect(screen.getByTestId('alert-banner')).toBeInTheDocument();
    expect(screen.getByTestId('alert-lap')).toHaveTextContent('Lap 3'); // first ALERT lap
    expect(screen.getByTestId('alert-laps-in-alert')).toHaveTextContent('2 laps in alert');
  });

  it('does NOT render before an ALERT lap is reached (nominal monitoring)', () => {
    render(<AlertBanner laps={LAPS} reachedIndex={1} />); // up to WATCH
    expect(screen.queryByTestId('alert-banner')).toBeNull();
  });

  it('renders the deferred alert-detail note (confidence/drivers/pit-window seam)', () => {
    render(<AlertBanner laps={LAPS} reachedIndex={3} />);
    expect(screen.getByTestId('alert-detail-note')).toHaveTextContent(/alert-payload seam/i);
  });

  it('uses the singular "lap" for a single lap in alert', () => {
    render(<AlertBanner laps={LAPS} reachedIndex={2} />);
    expect(screen.getByTestId('alert-laps-in-alert')).toHaveTextContent('1 lap in alert');
  });
});
