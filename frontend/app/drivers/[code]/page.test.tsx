import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: React.ReactNode; href: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));
vi.mock('next/navigation', () => ({ useRouter: () => ({ refresh: vi.fn() }) }));
vi.mock('@/lib/api/endpoints', () => ({
  getFingerprint: vi.fn(),
  getDrivers: vi.fn(),
  getSetup: vi.fn(),
  getStyleSpace: vi.fn(),
}));

import { getDrivers, getFingerprint, getSetup, getStyleSpace } from '@/lib/api/endpoints';

import DriverProfilePage from './page';

const fingerprint = {
  ok: true as const,
  status: 200,
  partial: false,
  flags: [],
  data: {
    driver_id: 'VER',
    vector_5d: [0.1, 0.2, 0.3, 0.4, 0.5],
    shap_radar: { throttle: 0.9, braking: 0.3, speed: 0.7, gear: 0.4, engine: 0.6, cornering: 0.5 },
    shap_signed: { F16: 0.455, F02: 0.317 },
    nl_description: 'Aggressive on throttle.',
    stability: {
      driver_id: 'VER',
      stability_score: 0.88,
      pairwise_js: { '2022-2023': 0.02 },
      seasons_available: 3,
      partial: false,
    },
  },
};
const drivers = {
  ok: true as const,
  status: 200,
  partial: false,
  flags: [],
  data: [
    {
      id: 'VER',
      code: 'VER',
      team: 'Red Bull',
      full_name: 'Max Verstappen',
      radar_values: fingerprint.data.shap_radar,
      signature: [{ feature: 'F02', attribution: 1.14, percentile: 1.0 }],
      profile: {
        laps_analysed: 4116,
        seasons: [2022, 2025],
        n_circuits: 23,
        lap_time_consistency_s: 1.14,
        identification_f1: 0.53,
      },
    },
  ],
};
const setup = {
  ok: true as const,
  status: 200,
  partial: false,
  flags: [],
  data: {
    driver: 'VER',
    circuit: 'Bahrain',
    parameters: [
      {
        name: 'front_wing_angle',
        kind: 'DEGREES',
        value: 6,
        value_range: null,
        category: null,
        confidence: 0.9,
        rating: 'HIGH',
        unit: 'degrees',
        fia_min: 0,
        fia_max: 12,
        constraint_flags: [],
        shap_top_features: [],
        shap_available: true,
      },
    ],
    consistency_warning: false,
    monaco_override: false,
    approximation_disclaimer: 'Setup targets are approximated from telemetry, not measured.',
    model_version: 'v',
  },
};
const styleSpace = {
  ok: true as const,
  status: 200,
  partial: false,
  flags: [],
  data: {
    drivers: [
      { id: 'VER', code: 'VER', coords_2d: [1, 2], coords_5d: [0, 0, 0, 0, 0], season: null },
      { id: 'LEC', code: 'LEC', coords_2d: [3, 1], coords_5d: [0, 0, 0, 0, 0], season: null },
    ],
  },
};

beforeEach(() => {
  for (const fn of [getFingerprint, getDrivers, getSetup, getStyleSpace]) {
    (fn as Mock).mockReset();
  }
  (getFingerprint as Mock).mockResolvedValue(fingerprint);
  (getDrivers as Mock).mockResolvedValue(drivers);
  (getSetup as Mock).mockResolvedValue(setup);
  (getStyleSpace as Mock).mockResolvedValue(styleSpace);
});

describe('DriverProfilePage (Screen 2)', () => {
  it('renders header, style profile, signal weights, facts, setup and style map', async () => {
    render(await DriverProfilePage({ params: { code: 'VER' } }));

    expect(screen.getByTestId('profile-header')).toHaveTextContent('Max Verstappen');
    expect(screen.getByTestId('style-profile-card')).toBeInTheDocument();
    expect(screen.getByTestId('signal-weights-card')).toBeInTheDocument();
    expect(screen.getByTestId('driver-summary')).toHaveTextContent('Aggressive on throttle.');
    expect(screen.getByTestId('facts-card')).toHaveTextContent('4,116');
    expect(screen.getByTestId('setup-card')).toBeInTheDocument();
    expect(screen.getByTestId('setup-param-front_wing_angle')).toHaveTextContent('6.0 degrees');
    expect(screen.getByTestId('style-map-card')).toBeInTheDocument();
    expect(screen.getByTestId('style-dot-VER')).toHaveAttribute('data-highlight');
  });

  it('renders a PageError when the driver is not found (404)', async () => {
    (getFingerprint as Mock).mockResolvedValue({
      ok: false,
      code: 'NOT_FOUND',
      detail: 'nope',
      status: 404,
    });

    render(await DriverProfilePage({ params: { code: 'ZZZ' } }));
    expect(screen.getByTestId('error-card')).toBeInTheDocument();
    expect(screen.getByTestId('error-code')).toHaveTextContent('NOT_FOUND');
  });

  it('shows an ErrorCard (not "no data") when setup is a real outage — HIGH-3', async () => {
    (getSetup as Mock).mockResolvedValue({
      ok: false,
      code: 'DB_UNAVAILABLE',
      detail: 'db down',
      status: 503,
    });

    render(await DriverProfilePage({ params: { code: 'VER' } }));
    expect(screen.getByTestId('error-code')).toHaveTextContent('DB_UNAVAILABLE');
    expect(screen.queryByText(/No setup data/)).toBeNull();
  });
});
