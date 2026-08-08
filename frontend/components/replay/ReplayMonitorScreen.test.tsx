import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ getReplay: vi.fn(), getSessions: vi.fn() }));

import { getReplay, getSessions } from '@/lib/api/endpoints';
import type { DriverSummary, SessionSummary } from '@/lib/api/types';
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
  { id: 'VER', code: 'VER', team: 'Red Bull Racing' },
  { id: 'HAM', code: 'HAM', team: 'Mercedes' },
] as unknown as DriverSummary[];

const SESSIONS: SessionSummary[] = [
  {
    id: 'sess-bahrain',
    circuit: 'Bahrain',
    circuit_type: 'high_speed',
    season: 2023,
    round: 1,
    session_type: 'R',
    drivers: [
      { code: 'VER', n_laps: 4 },
      { code: 'HAM', n_laps: 3 },
    ],
  },
  {
    id: 'sess-monaco',
    circuit: 'Monaco',
    circuit_type: 'street',
    season: 2024,
    round: 6,
    session_type: 'R',
    drivers: [{ code: 'HAM', n_laps: 2 }],
  },
];

const ALERT = {
  alert_lap: 4,
  first_crossing_lap: 3,
  laps_since_threshold_crossed: 2,
  deviation_score: 0.95,
  threshold: 0.7,
  alert_confidence: 'HIGH',
  feature_attribution: { F27: 0.6 },
  top_features: ['F27'],
  optimal_window_remaining_laps: 5,
};

const REPLAY = {
  laps: [
    { lap_number: 1, feature_vector: {}, alert_state: 'BASELINE', deviation_score: null, fingerprint_snapshot: null, threshold: null, alert: null },
    { lap_number: 2, feature_vector: {}, alert_state: 'NOMINAL', deviation_score: 0.2, fingerprint_snapshot: { confidence_state: 'HIGH', coords_2d: [0, 1] }, threshold: 0.7, alert: null },
    { lap_number: 3, feature_vector: {}, alert_state: 'WATCH', deviation_score: 0.6, fingerprint_snapshot: null, threshold: 0.7, alert: null },
    { lap_number: 4, feature_vector: {}, alert_state: 'ALERT', deviation_score: 0.95, fingerprint_snapshot: null, threshold: 0.7, alert: ALERT },
  ],
};

beforeEach(() => useReplay.getState().reset());
afterEach(() => {
  vi.clearAllMocks();
  useReplay.getState().reset();
});

async function renderWithCatalog(replayResult: unknown, drivers: DriverSummary[] = DRIVERS) {
  (getSessions as Mock).mockResolvedValue(ok(SESSIONS));
  (getReplay as Mock).mockResolvedValue(replayResult);
  render(<ReplayMonitorScreen initialDrivers={drivers} />);
  await waitFor(() => expect(screen.getByTestId('session-chips')).toBeInTheDocument());
}

describe('ReplayMonitorScreen — session catalog (div #77 closed)', () => {
  it('lists sessions as chips, auto-selects the first and auto-loads its first driver', async () => {
    await renderWithCatalog(ok(REPLAY));
    expect(screen.getByTestId('session-chip-sess-bahrain')).toHaveAttribute('data-selected');
    expect(screen.getByTestId('driver-chip-VER')).toHaveAttribute('data-selected');
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(getReplay).toHaveBeenCalledWith('sess-bahrain', 'VER');
    // No manual UUID input in catalog mode.
    expect(screen.queryByTestId('session-input')).toBeNull();
  });

  it('clicking a driver chip loads that driver', async () => {
    await renderWithCatalog(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('driver-chip-HAM'));
    await waitFor(() => expect(getReplay).toHaveBeenCalledWith('sess-bahrain', 'HAM'));
  });

  it('switching session keeps the driver when they ran it, else falls to its first driver', async () => {
    await renderWithCatalog(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('session-chip-sess-monaco')); // VER did not run Monaco
    await waitFor(() => expect(getReplay).toHaveBeenCalledWith('sess-monaco', 'HAM'));
  });

  it('an empty catalog is honest absence — a note, no fetch, no error', async () => {
    (getSessions as Mock).mockResolvedValue(ok([]));
    render(<ReplayMonitorScreen initialDrivers={DRIVERS} />);
    await waitFor(() => expect(screen.getByTestId('catalog-empty')).toBeInTheDocument());
    expect(getReplay).not.toHaveBeenCalled();
  });
});

describe('ReplayMonitorScreen — catalog-unavailable fallback (manual UUID entry)', () => {
  async function renderFallback(drivers: DriverSummary[] = DRIVERS) {
    (getSessions as Mock).mockResolvedValue(err('DB_UNAVAILABLE', 503));
    render(<ReplayMonitorScreen initialDrivers={drivers} />);
    await waitFor(() => expect(screen.getByTestId('catalog-fallback')).toBeInTheDocument());
  }

  it('surfaces the catalog error and falls back to manual session entry', async () => {
    await renderFallback();
    expect(screen.getByTestId('error-code')).toHaveTextContent('DB_UNAVAILABLE');
    expect(screen.getByTestId('catalog-unavailable-note')).toBeInTheDocument();
    expect(screen.getByTestId('session-input')).toBeInTheDocument();
    expect(screen.getByTestId('driver-select')).toBeInTheDocument();
    expect(screen.getByTestId('load-replay')).toBeDisabled(); // until a session id is typed
  });

  it('loads a replay from a manually entered session id', async () => {
    await renderFallback();
    (getReplay as Mock).mockResolvedValue(ok(REPLAY));
    fireEvent.change(screen.getByTestId('session-input'), { target: { value: 'sess-123' } });
    fireEvent.click(screen.getByTestId('load-replay'));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(getReplay).toHaveBeenCalledWith('sess-123', 'VER');
  });

  it('falls back to a driver UUID input when /drivers returned nothing', async () => {
    await renderFallback([]);
    expect(screen.getByTestId('driver-input')).toBeInTheDocument();
  });

  it('grid mode with no /drivers → note + disabled Load (honest, never a blank grid)', async () => {
    await renderFallback([]);
    fireEvent.click(screen.getByTestId('mode-grid'));
    fireEvent.change(screen.getByTestId('session-input'), { target: { value: 'sess-1' } });
    expect(screen.getByTestId('grid-no-drivers-note')).toBeInTheDocument();
    expect(screen.getByTestId('load-replay')).toBeDisabled();
  });
});

describe('ReplayMonitorScreen — monitor + playback', () => {
  it('renders the full monitor: gauge, timeline, playback, drift panel, status', async () => {
    await renderWithCatalog(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.getByTestId('deviation-gauge')).toBeInTheDocument();
    expect(screen.getByTestId('deviation-timeline')).toBeInTheDocument();
    expect(screen.getByTestId('playback-controls')).toBeInTheDocument();
    expect(screen.getByTestId('fingerprint-drift')).toBeInTheDocument();
    expect(screen.getByTestId('monitor-status')).toBeInTheDocument();
  });

  it('gauge shows "—" on the null baseline lap, then the score after scrubbing', async () => {
    await renderWithCatalog(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.getByTestId('deviation-gauge-value')).toHaveTextContent('—');
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '1' } });
    expect(screen.getByTestId('deviation-gauge-value')).toHaveTextContent('0.20');
  });

  it('WATCH lap → amber gauge tone, still no banner (alert not fired yet)', async () => {
    await renderWithCatalog(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '2' } });
    expect(screen.getByTestId('deviation-gauge')).toHaveAttribute('data-tone', 'watch');
    expect(screen.queryByTestId('alert-banner')).toBeNull();
  });

  it('the served alert banner appears on the ALERT lap with the real payload', async () => {
    await renderWithCatalog(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.queryByTestId('alert-banner')).toBeNull(); // starts on the BASELINE lap
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '3' } });
    expect(screen.getByTestId('alert-banner')).toBeInTheDocument();
    expect(screen.getByTestId('alert-lap')).toHaveTextContent('Lap 4');
    expect(screen.getByTestId('alert-confidence')).toHaveTextContent('HIGH confidence');
    expect(screen.getByTestId('alert-pit-window')).toHaveTextContent('5 laps remaining');
  });
});

describe('ReplayMonitorScreen — replay failure fixtures (§6)', () => {
  it('503 HEALTH_FAIL → retryable ErrorCard (fresh checkout)', async () => {
    await renderWithCatalog(err('HEALTH_FAIL', 503, 'no session data loaded'));
    await waitFor(() => expect(screen.getByTestId('error-card')).toBeInTheDocument());
    expect(screen.getByTestId('error-code')).toHaveTextContent('HEALTH_FAIL');
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('404 NOT_FOUND → EmptyState (honest absence)', async () => {
    await renderWithCatalog(err('NOT_FOUND', 404));
    await waitFor(() => expect(screen.getByTestId('empty-state')).toBeInTheDocument());
  });

  it('206 → renders the monitor with a partial notice', async () => {
    await renderWithCatalog(ok(REPLAY, { partial: true }));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    expect(screen.getByTestId('partial-notice')).toBeInTheDocument();
  });

  it('empty laps → EmptyState', async () => {
    await renderWithCatalog(ok({ laps: [] }));
    await waitFor(() => expect(screen.getByTestId('empty-state')).toBeInTheDocument());
  });

  it('UI_NETWORK → ErrorCard', async () => {
    await renderWithCatalog(err('UI_NETWORK', 0));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('UI_NETWORK'));
  });

  it('UI_TIMEOUT → ErrorCard', async () => {
    await renderWithCatalog(err('UI_TIMEOUT', 0));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('UI_TIMEOUT'));
  });

  it('UI_SCHEMA_MISMATCH → ErrorCard', async () => {
    await renderWithCatalog(err('UI_SCHEMA_MISMATCH', 200));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('UI_SCHEMA_MISMATCH'));
  });
});

describe('ReplayMonitorScreen — all-drivers grid mode (div #76)', () => {
  it('switching to grid mode fans out over the SESSION’s replayable drivers', async () => {
    await renderWithCatalog(ok(REPLAY));
    await waitFor(() => expect(screen.getByTestId('replay-monitor')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('mode-grid'));
    await waitFor(() => expect(screen.getByTestId('driver-grid')).toBeInTheDocument());
    // Bahrain has two replayable drivers (VER + HAM) → two cells, keyed by code.
    expect(screen.getAllByTestId('grid-cell')).toHaveLength(2);
    expect(screen.queryByTestId('driver-chips')).toBeNull(); // single-driver picker hidden
  });
});
