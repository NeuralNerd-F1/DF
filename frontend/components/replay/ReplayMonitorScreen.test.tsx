import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ getReplay: vi.fn() }));

import { getReplay } from '@/lib/api/endpoints';
import type { DriverSummary } from '@/lib/api/types';
import { useReplay } from '@/lib/state/replay';

import { ReplayMonitorScreen } from './ReplayMonitorScreen';

const ok = (data: unknown, opts: { partial?: boolean } = {}) => ({
  ok: true,
  data,
  status: opts.partial ? 206 : 200,
  partial: Boolean(opts.partial),
  flags: opts.partial ? ['REPLAY_PARTIAL'] : [],
});
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const DRIVERS = [
  { id: 'uuid-ver', code: 'VER' },
  { id: 'uuid-ham', code: 'HAM' },
] as unknown as DriverSummary[];

const REPLAY = {
  laps: [
    { lap_number: 1, feature_vector: {}, alert_state: 'BASELINE', deviation_score: null, fingerprint_snapshot: null },
    { lap_number: 2, feature_vector: {}, alert_state: 'NOMINAL', deviation_score: 0.2, fingerprint_snapshot: { confidence_state: 'HIGH' } },
    { lap_number: 3, feature_vector: {}, alert_state: 'WATCH', deviation_score: 0.6, fingerprint_snapshot: null },
    { lap_number: 4, feature_vector: {}, alert_state: 'ALERT', deviation_score: 0.95, fingerprint_snapshot: null },
  ],
};

beforeEach(() => useReplay.getState().reset());
afterEach(() => {
  vi.clearAllMocks();
  useReplay.getState().reset();
});

async function loadWith(result: unknown, drivers: DriverSummary[] = DRIVERS) {
  (getReplay as Mock).mockResolvedValue(result);
  render(<ReplayMonitorScreen initialDrivers={drivers} />);
  fireEvent.change(screen.getByTestId('session-input'), { target: { value: 'sess-123' } });
  fireEvent.click(screen.getByTestId('load-replay'));
}

describe('ReplayMonitorScreen — selection (div #77)', () => {
  it('lists drivers from /drivers and takes a session UUID (no catalog)', () => {
    render(<ReplayMonitorScreen initialDrivers={DRIVERS} />);
    expect(screen.getByTestId('driver-select')).toBeInTheDocument();
    expect(screen.getByTestId('session-input')).toBeInTheDocument();
    expect(screen.getByTestId('no-catalog-note')).toBeInTheDocument();
  });

  it('falls back to a driver UUID input when /drivers returned nothing', () => {
    render(<ReplayMonitorScreen initialDrivers={[]} />);
    expect(screen.getByTestId('driver-input')).toBeInTheDocument();
  });

  it('disables Load until a session id is entered', () => {
    render(<ReplayMonitorScreen initialDrivers={DRIVERS} />);
    expect(screen.getByTestId('load-replay')).toBeDisabled();
  });
});

describe('ReplayMonitorScreen — success + playback', () => {
  it('renders the full monitor on a successful replay', async () => {
    await loadWith(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.getByTestId('deviation-gauge')).toBeInTheDocument();
    expect(screen.getByTestId('deviation-timeline')).toBeInTheDocument();
    expect(screen.getByTestId('playback-controls')).toBeInTheDocument();
    expect(screen.getByTestId('per-lap-radar')).toBeInTheDocument();
  });

  it('gauge shows "—" on the null baseline lap, then the score after scrubbing', async () => {
    await loadWith(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.getByTestId('deviation-gauge-value')).toHaveTextContent('—');
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '1' } });
    expect(screen.getByTestId('deviation-gauge-value')).toHaveTextContent('0.20');
  });

  it('WATCH lap → amber gauge tone, still no banner', async () => {
    await loadWith(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '2' } });
    expect(screen.getByTestId('deviation-gauge')).toHaveAttribute('data-tone', 'watch');
    expect(screen.queryByTestId('alert-banner')).toBeNull();
  });

  it('the alert banner appears after scrubbing to the ALERT lap', async () => {
    await loadWith(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.queryByTestId('alert-banner')).toBeNull(); // starts on the BASELINE lap
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '3' } });
    expect(screen.getByTestId('alert-banner')).toBeInTheDocument();
    expect(screen.getByTestId('alert-lap')).toHaveTextContent('Lap 4');
  });
});

describe('ReplayMonitorScreen — failure fixtures (§6)', () => {
  it('503 HEALTH_FAIL → retryable ErrorCard (fresh checkout)', async () => {
    await loadWith(err('HEALTH_FAIL', 503, 'no session data loaded'));
    await waitFor(() => expect(screen.getByTestId('error-card')).toBeInTheDocument());
    expect(screen.getByTestId('error-code')).toHaveTextContent('HEALTH_FAIL');
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('503 DB_UNAVAILABLE → retryable ErrorCard', async () => {
    await loadWith(err('DB_UNAVAILABLE', 503));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('DB_UNAVAILABLE'));
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('404 NOT_FOUND → EmptyState (honest absence)', async () => {
    await loadWith(err('NOT_FOUND', 404));
    await waitFor(() => expect(screen.getByTestId('empty-state')).toBeInTheDocument());
  });

  it('206 → renders the monitor with a partial notice', async () => {
    await loadWith(ok(REPLAY, { partial: true }));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.getByTestId('partial-notice')).toBeInTheDocument();
  });

  it('empty laps → EmptyState', async () => {
    await loadWith(ok({ laps: [] }));
    await waitFor(() => expect(screen.getByTestId('empty-state')).toBeInTheDocument());
  });

  it('UI_NETWORK → ErrorCard', async () => {
    await loadWith(err('UI_NETWORK', 0));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('UI_NETWORK'));
  });

  it('UI_TIMEOUT → ErrorCard', async () => {
    await loadWith(err('UI_TIMEOUT', 0));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('UI_TIMEOUT'));
  });

  it('UI_SCHEMA_MISMATCH → ErrorCard', async () => {
    await loadWith(err('UI_SCHEMA_MISMATCH', 200));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('UI_SCHEMA_MISMATCH'));
  });
});

describe('ReplayMonitorScreen — 8-driver grid mode (div #76, F19b)', () => {
  it('switching to grid mode hides the single-driver select', () => {
    render(<ReplayMonitorScreen initialDrivers={DRIVERS} />);
    expect(screen.getByTestId('driver-select')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('mode-grid'));
    expect(screen.queryByTestId('driver-select')).toBeNull();
  });

  it('loads the N-parallel fan-out grid (one cell per driver)', async () => {
    (getReplay as Mock).mockResolvedValue(ok(REPLAY));
    render(<ReplayMonitorScreen initialDrivers={DRIVERS} />);
    fireEvent.click(screen.getByTestId('mode-grid'));
    fireEvent.change(screen.getByTestId('session-input'), { target: { value: 'sess-1' } });
    fireEvent.click(screen.getByTestId('load-replay'));
    await waitFor(() => expect(screen.getByTestId('driver-grid')).toBeInTheDocument());
    expect(screen.getAllByTestId('grid-cell')).toHaveLength(DRIVERS.length);
  });

  it('grid mode with no /drivers → note + disabled Load (honest, never a blank grid)', () => {
    render(<ReplayMonitorScreen initialDrivers={[]} />);
    fireEvent.click(screen.getByTestId('mode-grid'));
    fireEvent.change(screen.getByTestId('session-input'), { target: { value: 'sess-1' } });
    expect(screen.getByTestId('grid-no-drivers-note')).toBeInTheDocument();
    expect(screen.getByTestId('load-replay')).toBeDisabled();
  });
});
