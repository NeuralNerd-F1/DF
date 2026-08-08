import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ getReplay: vi.fn() }));

import { getReplay } from '@/lib/api/endpoints';
import { useReplay } from '@/lib/state/replay';

import { DriverReplayGrid, type GridDriver } from './DriverReplayGrid';

const ok = (data: unknown, opts: { partial?: boolean } = {}) => ({
  ok: true,
  data,
  status: opts.partial ? 206 : 200,
  partial: Boolean(opts.partial),
  flags: opts.partial ? ['REPLAY_PARTIAL'] : [],
});
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const DRIVERS: GridDriver[] = [
  { code: 'VER', team: 'Red Bull Racing' },
  { code: 'HAM', team: 'Mercedes' },
  { code: 'LEC', team: 'Ferrari' },
];

const ALERT = {
  alert_lap: 2,
  first_crossing_lap: 2,
  laps_since_threshold_crossed: 1,
  deviation_score: 0.9,
  threshold: 0.7,
  alert_confidence: 'MEDIUM',
  feature_attribution: { F27: 0.5 },
  top_features: ['F27'],
  optimal_window_remaining_laps: null,
};

const REPLAY = (withAlert = false) => ({
  laps: [
    {
      lap_number: 1,
      feature_vector: {},
      alert_state: 'NOMINAL',
      deviation_score: 0.2,
      fingerprint_snapshot: null,
      threshold: 0.7,
      alert: null,
    },
    {
      lap_number: 2,
      feature_vector: {},
      alert_state: withAlert ? 'ALERT' : 'WATCH',
      deviation_score: withAlert ? 0.9 : 0.6,
      fingerprint_snapshot: null,
      threshold: 0.7,
      alert: withAlert ? ALERT : null,
    },
  ],
});

beforeEach(() => useReplay.getState().reset());
afterEach(() => {
  vi.clearAllMocks();
  useReplay.getState().reset();
});

describe('DriverReplayGrid (div #76 — N-parallel fan-out)', () => {
  it('fans out one getReplay per driver CODE with the shared session id', async () => {
    (getReplay as Mock).mockResolvedValue(ok(REPLAY()));
    render(<DriverReplayGrid drivers={DRIVERS} sessionId="sess-1" />);
    expect(screen.getAllByTestId('grid-cell')).toHaveLength(3);
    await waitFor(() => expect(getReplay).toHaveBeenCalledTimes(3));
    expect(getReplay).toHaveBeenCalledWith('sess-1', 'VER');
    expect(getReplay).toHaveBeenCalledWith('sess-1', 'HAM');
    expect(getReplay).toHaveBeenCalledWith('sess-1', 'LEC');
  });

  it('renders each cell as an independent four-state panel', async () => {
    (getReplay as Mock).mockImplementation((_s: string, code: string) => {
      if (code === 'VER') return Promise.resolve(ok(REPLAY()));
      if (code === 'HAM') return Promise.resolve(err('NOT_FOUND', 404));
      return Promise.resolve(err('HEALTH_FAIL', 503, 'no data'));
    });
    render(<DriverReplayGrid drivers={DRIVERS} sessionId="s" />);
    await waitFor(() => expect(screen.getByTestId('deviation-gauge')).toBeInTheDocument()); // VER ok
    expect(screen.getByTestId('empty-state')).toBeInTheDocument(); // HAM 404 → absence
    expect(screen.getByTestId('error-code')).toHaveTextContent('HEALTH_FAIL'); // LEC 503 → error
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('shares ONE playback position; scrubbing shows the served alert chip on every cell', async () => {
    (getReplay as Mock).mockResolvedValue(ok(REPLAY(true)));
    render(<DriverReplayGrid drivers={DRIVERS} sessionId="s" />);
    await waitFor(() => expect(screen.getAllByTestId('deviation-gauge')).toHaveLength(3));
    expect(screen.queryAllByTestId('cell-alert')).toHaveLength(0); // starts on the NOMINAL lap
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '1' } }); // reach the ALERT lap
    const chips = screen.getAllByTestId('cell-alert');
    expect(chips).toHaveLength(3);
    expect(chips[0]).toHaveTextContent('lap 2');
    expect(chips[0]).toHaveTextContent('MEDIUM'); // served confidence, not client-derived
  });

  it('retries a single failed cell independently', async () => {
    const seen: string[] = [];
    (getReplay as Mock).mockImplementation((_s: string, code: string) => {
      seen.push(code);
      if (seen.filter((x) => x === code).length === 1) return Promise.resolve(err('HEALTH_FAIL', 503));
      return Promise.resolve(ok(REPLAY()));
    });
    render(<DriverReplayGrid drivers={[{ code: 'HAM' }]} sessionId="s" />);
    await waitFor(() => expect(screen.getByTestId('error-card')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.getByTestId('deviation-gauge')).toBeInTheDocument());
  });

  it('marks a 206 cell as a partial replay', async () => {
    (getReplay as Mock).mockResolvedValue(ok(REPLAY(), { partial: true }));
    render(<DriverReplayGrid drivers={[{ code: 'VER' }]} sessionId="s" />);
    await waitFor(() => expect(screen.getByTestId('cell-partial')).toBeInTheDocument());
  });
});
