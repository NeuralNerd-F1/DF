import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ReplayLap } from '@/lib/api/types';

import { DeviationTimeline } from './DeviationTimeline';

const lap = (n: number, state: string, score: number | null, threshold: number | null = 0.7): ReplayLap =>
  ({
    lap_number: n,
    feature_vector: {},
    alert_state: state,
    deviation_score: score,
    fingerprint_snapshot: null,
    threshold,
    alert: null,
  }) as ReplayLap;

const LAPS: ReplayLap[] = [
  lap(1, 'NOMINAL', 0.1),
  lap(2, 'BASELINE', null, null), // gap
  lap(3, 'WATCH', 0.5),
  lap(4, 'ALERT', 0.9),
];

describe('DeviationTimeline (div #73 closed — real τ line)', () => {
  it('plots reached laps, highlights ALERT, and renders null scores as gaps', () => {
    render(<DeviationTimeline laps={LAPS} reachedIndex={3} />);
    expect(screen.getByTestId('deviation-timeline')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-alert-point')).toBeInTheDocument();
    // 3 non-null scores among 4 reached laps (the null lap is a gap, not a 0 point)
    expect(screen.getAllByTestId(/timeline-(alert-)?point/)).toHaveLength(3);
  });

  it('only draws laps up to the reached frontier', () => {
    render(<DeviationTimeline laps={LAPS} reachedIndex={0} />);
    expect(screen.getAllByTestId('timeline-point')).toHaveLength(1);
    expect(screen.queryByTestId('timeline-alert-point')).toBeNull();
  });

  it('draws the REAL dashed τ threshold line with its value', () => {
    render(<DeviationTimeline laps={LAPS} reachedIndex={3} />);
    expect(screen.getByTestId('timeline-threshold-line')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-threshold-line')).toHaveTextContent('τ 0.70');
    expect(screen.getByTestId('timeline-caption')).toHaveTextContent(/dashed = alert threshold/i);
  });

  it('omits the line and says so when τ is unavailable — never a fabricated line', () => {
    render(<DeviationTimeline laps={[lap(1, 'BASELINE', null, null)]} reachedIndex={0} />);
    expect(screen.queryByTestId('timeline-threshold-line')).toBeNull();
    expect(screen.getByTestId('timeline-caption')).toHaveTextContent(/τ unavailable/i);
  });

  it('renders no points when every reached score is null', () => {
    render(<DeviationTimeline laps={[lap(1, 'BASELINE', null, null)]} reachedIndex={0} />);
    expect(screen.queryByTestId('timeline-point')).toBeNull();
    expect(screen.queryByTestId('timeline-alert-point')).toBeNull();
  });
});
