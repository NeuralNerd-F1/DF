import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ getCompare: vi.fn(), getSetup: vi.fn() }));

import { getCompare, getSetup } from '@/lib/api/endpoints';
import { CIRCUITS } from '@/lib/design/circuits';

import { ComparisonScreen } from './ComparisonScreen';

const ok = (data: unknown) => ({ ok: true, data, status: 200, partial: false, flags: [] });
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const DRIVERS = [
  { id: 'VER', code: 'VER', team: 'Red Bull', radar_values: {} },
  { id: 'HAM', code: 'HAM', team: 'Mercedes', radar_values: {} },
];

const CMP = ok({
  driver_1_radar: { throttle: 0.6, braking: 0.5 },
  driver_2_radar: { throttle: 0.4, braking: 0.7 },
  delta_shap: { throttle: 0.2, braking: -0.2 },
  most_distinctive_features: ['throttle', 'braking'],
  compatibility_delta: 0.4,
});

const SETUP = ok({
  driver: 'X',
  circuit: 'Bahrain',
  parameters: [{ name: 'front_wing', value: 7, kind: 'NUM', confidence: 0.8, rating: 'HIGH' }],
  approximation_disclaimer: '',
  model_version: 'v1',
});

afterEach(() => vi.clearAllMocks());

describe('ComparisonScreen (master §10.5)', () => {
  it('renders two radars, a delta radar, distinctive features, compat and setup delta', async () => {
    (getCompare as Mock).mockResolvedValue(CMP);
    (getSetup as Mock).mockResolvedValue(SETUP);
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('delta-radar')).toBeInTheDocument());
    expect(screen.getAllByTestId('fingerprint-radar')).toHaveLength(2);
    expect(screen.getByTestId('distinctive-features')).toBeInTheDocument();
    expect(screen.getByTestId('compatibility-projection')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId('setup-delta-table')).toBeInTheDocument());
  });

  it('swaps drivers and re-fetches /compare with the swapped order (delta flips)', async () => {
    (getCompare as Mock).mockResolvedValue(CMP);
    (getSetup as Mock).mockResolvedValue(SETUP);
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(getCompare as Mock).toHaveBeenCalledWith('VER', 'HAM'));
    fireEvent.click(screen.getByTestId('swap-button'));
    await waitFor(() => expect(getCompare as Mock).toHaveBeenCalledWith('HAM', 'VER'));
  });

  it('renders a loading Skeleton before /compare resolves (four-state discipline)', () => {
    (getCompare as Mock).mockReturnValue(new Promise(() => {})); // never resolves
    (getSetup as Mock).mockReturnValue(new Promise(() => {}));
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    expect(screen.getByTestId('skeleton')).toBeInTheDocument();
  });

  it('re-fetches setups when the circuit changes', async () => {
    (getCompare as Mock).mockResolvedValue(CMP);
    (getSetup as Mock).mockResolvedValue(SETUP);
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('setup-delta-table')).toBeInTheDocument());
    (getSetup as Mock).mockClear();
    const other = CIRCUITS[1];
    fireEvent.change(screen.getByTestId('compare-circuit'), { target: { value: other } });
    await waitFor(() => expect(getSetup as Mock).toHaveBeenCalledWith('VER', other));
  });

  it('surfaces a 404 NOT_FOUND from /compare as an ErrorCard', async () => {
    (getCompare as Mock).mockResolvedValue(err('NOT_FOUND', 404, 'unknown driver'));
    (getSetup as Mock).mockResolvedValue(SETUP);
    render(<ComparisonScreen initialDrivers={DRIVERS} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('NOT_FOUND'));
  });
});
