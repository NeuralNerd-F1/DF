import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { TyreWearScreen } from './TyreWearScreen';

const api = vi.hoisted(() => ({ getTyres: vi.fn(), getDrivers: vi.fn() }));
vi.mock('@/lib/api/endpoints', () => api);
vi.mock('next/navigation', () => ({ useRouter: () => ({ refresh: vi.fn() }) }));

const CURVE = [
  { age: 1, delta_s: 3.4, q25: 2.8, q75: 4.0, n: 27 },
  { age: 2, delta_s: 0.0, q25: -0.2, q75: 0.2, n: 62 },
  { age: 10, delta_s: 0.4, q25: 0.1, q75: 0.8, n: 70 },
  { age: 20, delta_s: 0.9, q25: 0.4, q75: 1.4, n: 40 },
];

const STATS = {
  n_laps: 1657,
  n_stints: 83,
  median_stint_len: 19,
  max_stint_len: 32,
  deg_slope_s_per_lap: 0.062,
  total_delta_s: 1.12,
  fuel_slope_s_per_lap: -0.062,
  cliff_age: null,
  curve: CURVE,
  f27_median: 0.036,
  f28_median: null,
};

beforeEach(() => {
  vi.resetAllMocks();
  api.getTyres.mockResolvedValue({
    ok: true,
    data: {
      drivers: {
        VER: { MEDIUM: STATS, SOFT: { ...STATS, deg_slope_s_per_lap: 0.076, cliff_age: 24 } },
        LEC: { MEDIUM: { ...STATS, deg_slope_s_per_lap: 0.055 } },
      },
    },
    flags: [],
  });
  api.getDrivers.mockResolvedValue({
    ok: true,
    data: [{ id: 'VER', code: 'VER', team: 'Red Bull Racing', full_name: null, radar_values: {} }],
    flags: [],
  });
});

describe('TyreWearScreen', () => {
  it('renders the measured tiles for the first driver/compound (alphabetical default)', async () => {
    render(<TyreWearScreen />);
    await waitFor(() => expect(screen.getByTestId('driver-chips')).toBeInTheDocument());
    const root = screen.getByTestId('tyre-wear-screen');
    expect(screen.getByTestId('driver-chip-LEC')).toHaveAttribute('data-selected');
    expect(root).toHaveTextContent('19 laps');
    expect(root).toHaveTextContent('+0.055s/lap'); // LEC's slope — the alphabetical default
    expect(root).toHaveTextContent('+1.12s');
    expect(root).toHaveTextContent('—'); // no cliff detected → dash, not a made-up lap
  });

  it('switching driver and compound re-renders their measured stats', async () => {
    render(<TyreWearScreen />);
    await waitFor(() => expect(screen.getByTestId('driver-chip-VER')).toBeInTheDocument());
    await userEvent.click(screen.getByTestId('driver-chip-LEC'));
    expect(screen.getByTestId('tyre-wear-screen')).toHaveTextContent('+0.055s/lap');
    await userEvent.click(screen.getByTestId('driver-chip-VER'));
    await userEvent.click(screen.getByTestId('compound-chip-SOFT'));
    expect(screen.getByTestId('tyre-wear-screen')).toHaveTextContent('+0.076s/lap');
    expect(screen.getByTestId('tyre-wear-screen')).toHaveTextContent('Lap 24'); // SOFT's cliff
  });

  it('states the fuel-correction method under the chart', async () => {
    render(<TyreWearScreen />);
    await waitFor(() => expect(screen.getByTestId('degradation-chart')).toBeInTheDocument());
    expect(screen.getByText(/identified per race from its pit-stop structure/)).toBeInTheDocument();
    expect(screen.getByText(/throttle drift \+0\.036\/lap/)).toBeInTheDocument();
  });

  it('empty payload → explicit build hint, nothing invented', async () => {
    api.getTyres.mockResolvedValue({ ok: true, data: { drivers: {} }, flags: [] });
    render(<TyreWearScreen />);
    expect(await screen.findByTestId('no-tyre-data')).toBeInTheDocument();
  });

  it('a fetch failure surfaces the error code', async () => {
    api.getTyres.mockResolvedValue({ ok: false, code: 'API_ERROR', detail: 'down', flags: [] });
    render(<TyreWearScreen />);
    await waitFor(() => expect(screen.getByText(/API_ERROR/)).toBeInTheDocument());
  });
});
