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
}));

import { getDrivers, getFingerprint, getSetup } from '@/lib/api/endpoints';

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
  data: [{ id: 'VER', code: 'VER', team: 'Red Bull', radar_values: fingerprint.data.shap_radar }],
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
        category: 'aero',
        confidence: 0.9,
        rating: 'HIGH',
        unit: 'deg',
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

beforeEach(() => {
  (getFingerprint as Mock).mockReset();
  (getDrivers as Mock).mockReset();
  (getSetup as Mock).mockReset();
});

describe('DriverProfilePage (Screen 2)', () => {
  it('renders the profile: radar, NL, setup panel, multi-season, and honest per-lap/corner states', async () => {
    (getFingerprint as Mock).mockResolvedValue(fingerprint);
    (getDrivers as Mock).mockResolvedValue(drivers);
    (getSetup as Mock).mockResolvedValue(setup);

    render(await DriverProfilePage({ params: { code: 'VER' } }));

    expect(screen.getByTestId('profile-radar')).toBeInTheDocument();
    expect(screen.getByTestId('nl')).toHaveTextContent('Aggressive on throttle.');
    expect(screen.getByTestId('setup-panel')).toBeInTheDocument();
    expect(screen.getByTestId('multiseason-panel')).toHaveTextContent('0.88');
    // honest unavailable states (div #48 / #49)
    expect(screen.getByTestId('beeswarm-panel')).toHaveTextContent('not yet available');
    expect(screen.getByTestId('corner-tabs')).toHaveTextContent('not yet available');
  });

  it('renders a PageError when the driver is not found (404)', async () => {
    (getFingerprint as Mock).mockResolvedValue({ ok: false, code: 'NOT_FOUND', detail: 'nope', status: 404 });
    (getDrivers as Mock).mockResolvedValue(drivers);
    (getSetup as Mock).mockResolvedValue({ ok: false, code: 'NOT_FOUND', detail: '', status: 404 });

    render(await DriverProfilePage({ params: { code: 'ZZZ' } }));
    expect(screen.getByTestId('error-card')).toBeInTheDocument();
    expect(screen.getByTestId('error-code')).toHaveTextContent('NOT_FOUND');
  });

  it('shows an ErrorCard (not "no data") when setup is a real outage — HIGH-3', async () => {
    (getFingerprint as Mock).mockResolvedValue(fingerprint);
    (getDrivers as Mock).mockResolvedValue(drivers);
    (getSetup as Mock).mockResolvedValue({ ok: false, code: 'DB_UNAVAILABLE', detail: 'db down', status: 503 });

    render(await DriverProfilePage({ params: { code: 'VER' } }));
    expect(screen.getByTestId('error-code')).toHaveTextContent('DB_UNAVAILABLE');
    expect(screen.queryByText(/No setup data/)).toBeNull();
  });
});
