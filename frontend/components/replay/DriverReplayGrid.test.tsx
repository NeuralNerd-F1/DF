import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ getReplay: vi.fn() }));

import { getReplay } from '@/lib/api/endpoints';
import type { DriverSummary } from '@/lib/api/types';
import { useReplay } from '@/lib/state/replay';

import { DriverReplayGrid } from './DriverReplayGrid';

const ok = (data: unknown, opts: { partial?: boolean } = {}) => ({
  ok: true,
  data,
  status: opts.partial ? 206 : 200,
  partial: Boolean(opts.partial),
  flags: opts.partial ? ['REPLAY_PARTIAL'] : [],
});
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const DRIVERS = [
  { id: 'u-ver', code: 'VER' },
  { id: 'u-ham', code: 'HAM' },
  { id: 'u-lec', code: 'LEC' },
] as unknown as DriverSummary[];

const REPLAY = (withAlert = false) => ({
  laps: [
    { lap_number: 1, feature_vector: {}, alert_state: 'NOMINAL', deviation_score: 0.2, fingerprint_snapshot: null },
    {
      lap_number: 2,
      feature_vector: {},
      alert_state: withAlert ? 'ALERT' : 'WATCH',
      deviation_score: withAlert ? 0.9 : 0.6,
      fingerprint_snapshot: null,
    },
  ],
});

beforeEach(() => useReplay.getState().reset());
afterEach(() => {
  vi.clearAllMocks();
  useReplay.getState().reset();
});

describe('DriverReplayGrid (div #76 — N-parallel fan-out)', () => {
  it('fans out one getReplay per driver with the shared session id', async () => {
    (getReplay as Mock).mockResolvedValue(ok(REPLAY()));
    render(<DriverReplayGrid drivers={DRIVERS} sessionId="sess-1" />);
    expect(screen.getAllByTestId('grid-cell')).toHaveLength(3);
    await waitFor(() => expect(getReplay).toHaveBeenCalledTimes(3));
    expect(getReplay).toHaveBeenCalledWith('sess-1', 'u-ver');
    expect(getReplay).toHaveBeenCalledWith('sess-1', 'u-ham');
    expect(getReplay).toHaveBeenCalledWith('sess-1', 'u-lec');
  });

  it('renders each cell as an independent four-state panel', async () => {
    (getReplay as Mock).mockImplementation((_s: string, id: string) => {
      if (id === 'u-ver') return Promise.resolve(ok(REPLAY()));
      if (id === 'u-ham') return Promise.resolve(err('NOT_FOUND', 404));
      return Promise.resolve(err('HEALTH_FAIL', 503, 'no data'));
    });
    render(<DriverReplayGrid drivers={DRIVERS} sessionId="s" />);
    await waitFor(() => expect(screen.getByTestId('deviation-gauge')).toBeInTheDocument()); // VER ok
    expect(screen.getByTestId('empty-state')).toBeInTheDocument(); // HAM 404 → absence
    expect(screen.getByTestId('error-code')).toHaveTextContent('HEALTH_FAIL'); // LEC 503 → error
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('shares ONE playback position; scrubbing shows the ALERT chip on every cell', async () => {
    (getReplay as Mock).mockResolvedValue(ok(REPLAY(true)));
    render(<DriverReplayGrid drivers={DRIVERS} sessionId="s" />);
    await waitFor(() => expect(screen.getAllByTestId('deviation-gauge')).toHaveLength(3));
    expect(screen.queryAllByTestId('cell-alert')).toHaveLength(0); // starts on the NOMINAL lap
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '1' } }); // reach the ALERT lap
    expect(screen.getAllByTestId('cell-alert')).toHaveLength(3);
  });

  it('retries a single failed cell independently', async () => {
    const seen: string[] = [];
    (getReplay as Mock).mockImplementation((_s: string, id: string) => {
      seen.push(id);
      if (seen.filter((x) => x === id).length === 1) return Promise.resolve(err('HEALTH_FAIL', 503));
      return Promise.resolve(ok(REPLAY()));
    });
    render(<DriverReplayGrid drivers={[{ id: 'u-ham', code: 'HAM' }] as unknown as DriverSummary[]} sessionId="s" />);
    await waitFor(() => expect(screen.getByTestId('error-card')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.getByTestId('deviation-gauge')).toBeInTheDocument());
  });

  it('marks a 206 cell as a partial replay', async () => {
    (getReplay as Mock).mockResolvedValue(ok(REPLAY(), { partial: true }));
    render(<DriverReplayGrid drivers={[{ id: 'u-ver', code: 'VER' }] as unknown as DriverSummary[]} sessionId="s" />);
    await waitFor(() => expect(screen.getByTestId('cell-partial')).toBeInTheDocument());
  });
});
