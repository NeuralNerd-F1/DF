import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ReplayLap } from '@/lib/api/types';

import { PerLapRadarPanel } from './PerLapRadarPanel';

const withSnapshot = (snapshot: unknown): ReplayLap =>
  ({ lap_number: 1, feature_vector: {}, alert_state: 'NOMINAL', deviation_score: 0.1, fingerprint_snapshot: snapshot }) as ReplayLap;

describe('PerLapRadarPanel (div #75 — deferred, never fabricated)', () => {
  it('renders the deferred state and NO radar from 5D coords', () => {
    render(<PerLapRadarPanel lap={withSnapshot({ coords_5d: [1, 2, 3, 4, 5], coords_2d: [1, 2], confidence_state: 'HIGH' })} />);
    expect(screen.getByTestId('per-lap-radar-note')).toHaveTextContent(/not exposed/i);
    expect(screen.queryByTestId('fingerprint-radar')).toBeNull();
    expect(screen.queryByTestId('radar-polygon')).toBeNull();
  });

  it('surfaces the REAL per-lap confidence_state (and its badge on low confidence)', () => {
    render(<PerLapRadarPanel lap={withSnapshot({ confidence_state: 'LOW_CONFIDENCE' })} />);
    expect(screen.getByTestId('per-lap-confidence')).toHaveTextContent('LOW_CONFIDENCE');
    expect(screen.getByTestId('confidence-badge')).toBeInTheDocument();
  });

  it('shows only the deferred note when the snapshot is null', () => {
    render(<PerLapRadarPanel lap={withSnapshot(null)} />);
    expect(screen.getByTestId('per-lap-radar-note')).toBeInTheDocument();
    expect(screen.queryByTestId('per-lap-confidence')).toBeNull();
  });
});
