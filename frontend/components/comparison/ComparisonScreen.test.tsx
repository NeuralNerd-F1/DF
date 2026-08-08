import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({
  getCompare: vi.fn(),
  getFingerprint: vi.fn(),
  getSetup: vi.fn(),
}));

import { getCompare, getFingerprint, getSetup } from '@/lib/api/endpoints';
import type { DriverSummary } from '@/lib/api/types';
import { CIRCUITS } from '@/lib/design/circuits';

import { ComparisonScreen } from './ComparisonScreen';

const ok = (data: unknown) => ({ ok: true, data, status: 200, partial: false, flags: [] });
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const DRIVERS = [
  { id: 'VER', code: 'VER', team: 'Red Bull Racing', radar_values: {} },
  { id: 'HAM', code: 'HAM', team: 'Mercedes', radar_values: {} },
  { id: 'LEC', code: 'LEC', team: 'Ferrari', radar_values: {} },
] as unknown as DriverSummary[];

// Raw served magnitudes are UNBOUNDED (throttle 3.2) — the screen must scale, not clamp.
const CMP = ok({
  driver_1_radar: { throttle: 3.2, braking: 0.4 },
  driver_2_radar: { throttle: 1.6, braking: 0.8 },
  delta_shap: { throttle: 1.6, braking: -0.4 },
  most_distinctive_features: ['throttle', 'braking'],
  compatibility_delta: 2.0,
});

const FP = (signed: Record<string, number>) =>
  ok({
    driver_id: 'X',
    vector_5d: [],
    shap_radar: {},
    nl_description: '',
    stability: null,
    shap_signed: signed,
  });

const SETUP = ok({
  driver: 'X',
  circuit: 'Bahrain',
  parameters: [{ name: 'front_wing', value: 7, kind: 'NUM', confidence: 0.8, rating: 'HIGH' }],
  approximation_disclaimer: '',
  model_version: 'v1',
});

function mockAll() {
  (getCompare as Mock).mockResolvedValue(CMP);
  (getFingerprint as Mock).mockImplementation((code: string) =>
    Promise.resolve(FP(code === 'VER' ? { F01: 0.3, F06: 0.1 } : { F01: 0.1, F06: 0.4 })),
  );
  (getSetup as Mock).mockResolvedValue(SETUP);
}

afterEach(() => vi.clearAllMocks());

describe('ComparisonScreen (master §10.5, redesigned)', () => {
  it('renders two radars, the delta radar, distinctive features, compat and setup delta', async () => {
    mockAll();
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('delta-radar')).toBeInTheDocument());
    expect(screen.getAllByTestId('fingerprint-radar')).toHaveLength(2);
    expect(screen.getByTestId('radar-scale-note')).toBeInTheDocument();
    expect(screen.getByTestId('distinctive-features')).toBeInTheDocument();
    expect(screen.getByTestId('compatibility-projection')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId('setup-delta-table')).toBeInTheDocument());
  });

  it('uses the per-FEATURE signed-SHAP delta when both fingerprints serve it', async () => {
    mockAll();
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() =>
      expect(screen.getByTestId('distinctive-features')).toHaveAttribute('data-mode', 'feature'),
    );
    // F06 delta = 0.1 − 0.4 = −0.3 (largest |Δ|) → ranked first, leaning to driver 2.
    const rows = screen.getAllByTestId('distinctive-row');
    expect(rows[0]).toHaveAttribute('data-key', 'F06');
    expect(rows[0]).toHaveAttribute('data-sign', 'negative');
  });

  it('degrades to the per-domain delta (labelled) when a fingerprint fails — panel never dies', async () => {
    (getCompare as Mock).mockResolvedValue(CMP);
    (getFingerprint as Mock).mockResolvedValue(err('HEALTH_FAIL', 503));
    (getSetup as Mock).mockResolvedValue(SETUP);
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() =>
      expect(screen.getByTestId('distinctive-features')).toHaveAttribute('data-mode', 'domain'),
    );
    expect(screen.getByTestId('domain-fallback-note')).toBeInTheDocument();
  });

  it('picks drivers via chips and swap re-fetches /compare with the swapped order', async () => {
    mockAll();
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(getCompare as Mock).toHaveBeenCalledWith('VER', 'HAM'));
    fireEvent.click(screen.getByTestId('driver2-chip-LEC'));
    await waitFor(() => expect(getCompare as Mock).toHaveBeenCalledWith('VER', 'LEC'));
    fireEvent.click(screen.getByTestId('swap-button'));
    await waitFor(() => expect(getCompare as Mock).toHaveBeenCalledWith('LEC', 'VER'));
  });

  it('renders a loading Skeleton before /compare resolves (four-state discipline)', () => {
    (getCompare as Mock).mockReturnValue(new Promise(() => {})); // never resolves
    (getFingerprint as Mock).mockReturnValue(new Promise(() => {}));
    (getSetup as Mock).mockReturnValue(new Promise(() => {}));
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    expect(screen.getByTestId('skeleton')).toBeInTheDocument();
  });

  it('re-fetches setups when a circuit chip is picked', async () => {
    mockAll();
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('setup-delta-table')).toBeInTheDocument());
    (getSetup as Mock).mockClear();
    const other = CIRCUITS[1];
    fireEvent.click(screen.getByTestId(`circuit-chip-${other}`));
    await waitFor(() => expect(getSetup as Mock).toHaveBeenCalledWith('VER', other));
  });

  it('surfaces a /compare 404 as honest absence (EmptyState), other failures as ErrorCard', async () => {
    (getCompare as Mock).mockResolvedValue(err('NOT_FOUND', 404, 'no fingerprint'));
    (getFingerprint as Mock).mockResolvedValue(err('NOT_FOUND', 404));
    (getSetup as Mock).mockResolvedValue(SETUP);
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('empty-state')).toBeInTheDocument());

    (getCompare as Mock).mockResolvedValue(err('DB_UNAVAILABLE', 503));
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('DB_UNAVAILABLE'));
  });

  it('shows an EmptyState with fewer than two drivers', () => {
    mockAll();
    render(<ComparisonScreen initialDrivers={DRIVERS.slice(0, 1)} animate={false} />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });
});
