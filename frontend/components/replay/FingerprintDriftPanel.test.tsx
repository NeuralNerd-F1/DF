import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ReplayLap } from '@/lib/api/types';

import { FingerprintDriftPanel } from './FingerprintDriftPanel';

const lap = (n: number, snapshot: unknown): ReplayLap =>
  ({
    lap_number: n,
    feature_vector: {},
    alert_state: 'NOMINAL',
    deviation_score: 0.1,
    fingerprint_snapshot: snapshot,
    threshold: 0.7,
    alert: null,
  }) as ReplayLap;

const snap = (x: number, y: number, confidence: string | null = 'HIGH') => ({
  coords_5d: [x, y, 0, 0, 0],
  coords_2d: [x, y],
  confidence_state: confidence,
});

const LAPS: ReplayLap[] = [
  lap(1, snap(0, 0)),
  lap(2, null), // snapshot unavailable that lap — no point fabricated
  lap(3, snap(1, 2)),
  lap(4, snap(2, 4, 'OUT_OF_DISTRIBUTION')),
];

describe('FingerprintDriftPanel (master §10.7 — the fingerprint evolving)', () => {
  it('plots one REAL point per reached snapshot and emphasises the current lap', () => {
    render(<FingerprintDriftPanel laps={LAPS} reachedIndex={3} />);
    expect(screen.getByTestId('fingerprint-drift')).toBeInTheDocument();
    expect(screen.getAllByTestId('drift-point')).toHaveLength(2); // laps 1 & 3
    expect(screen.getByTestId('drift-current')).toBeInTheDocument(); // lap 4
    expect(screen.getByTestId('drift-trail')).toBeInTheDocument();
  });

  it('only draws up to the reached frontier', () => {
    render(<FingerprintDriftPanel laps={LAPS} reachedIndex={0} />);
    expect(screen.queryAllByTestId('drift-point')).toHaveLength(0);
    expect(screen.getByTestId('drift-current')).toBeInTheDocument(); // just lap 1
  });

  it('surfaces the REAL per-lap confidence_state with its badge', () => {
    render(<FingerprintDriftPanel laps={LAPS} reachedIndex={3} />);
    expect(screen.getByTestId('drift-confidence')).toHaveTextContent('OUT_OF_DISTRIBUTION');
    expect(screen.getByTestId('confidence-badge')).toBeInTheDocument();
  });

  it('renders an honest empty note when no lap carries a snapshot', () => {
    render(<FingerprintDriftPanel laps={[lap(1, null), lap(2, null)]} reachedIndex={1} />);
    expect(screen.getByTestId('drift-empty-note')).toBeInTheDocument();
    expect(screen.queryByTestId('drift-current')).toBeNull();
  });

  it('states that projection axes carry no units and no radar is fabricated', () => {
    render(<FingerprintDriftPanel laps={LAPS} reachedIndex={3} />);
    expect(screen.getByTestId('drift-caption')).toHaveTextContent(/no physical units/i);
    expect(screen.getByTestId('drift-caption')).toHaveTextContent(/radar is not exposed/i);
  });
});
