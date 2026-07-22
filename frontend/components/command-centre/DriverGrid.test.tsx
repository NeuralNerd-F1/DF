import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: React.ReactNode; href: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));
vi.mock('@/lib/api/endpoints', () => ({ getStability: vi.fn(), getSetup: vi.fn() }));

import { getSetup, getStability } from '@/lib/api/endpoints';
import { useCommandCentre } from '@/lib/state/commandCentre';

import { DriverGrid } from './DriverGrid';

function driver(code: string) {
  return {
    id: code,
    code,
    team: 'Ferrari',
    radar_values: { throttle: 0.5, braking: 0.3, speed: 0.7, gear: 0.4, engine: 0.6, cornering: 0.5 },
  };
}
const stab = (v: number) => ({
  ok: true as const,
  data: { driver_id: '', stability_score: v, pairwise_js: {}, seasons_available: 3, partial: false },
  status: 200,
  partial: false,
  flags: [],
});
const setup = (conf: number) => ({
  ok: true as const,
  data: {
    driver: '',
    circuit: '',
    parameters: [{ name: 'x', kind: 'DEGREES', confidence: conf, rating: 'HIGH' }],
    approximation_disclaimer: '',
    model_version: 'v',
  },
  status: 200,
  partial: false,
  flags: [],
});

beforeEach(() => {
  useCommandCentre.setState({ circuit: 'Bahrain', sortKey: 'compatibility' });
  (getStability as Mock).mockReset();
  (getSetup as Mock).mockReset();
});

describe('DriverGrid (master §10.2)', () => {
  it('renders the empty state on no drivers (200 + [])', () => {
    render(<DriverGrid initialDrivers={[]} animate={false} />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });

  it('renders a card per driver with fetched compatibility + stability', async () => {
    (getStability as Mock).mockResolvedValue(stab(0.9));
    (getSetup as Mock).mockResolvedValue(setup(0.7));
    render(<DriverGrid initialDrivers={[driver('VER'), driver('HAM')]} animate={false} />);
    expect(screen.getByTestId('driver-card-VER')).toBeInTheDocument();
    expect(screen.getByTestId('driver-card-HAM')).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByTestId('compat')[0]).toHaveTextContent('70%'));
    expect(screen.getAllByTestId('stability')[0]).toHaveTextContent('90%');
  });

  it('shows "—" for a compatibility 404 — no fabricated score (div #47)', async () => {
    (getStability as Mock).mockResolvedValue(stab(0.9));
    (getSetup as Mock).mockResolvedValue({ ok: false, code: 'NOT_FOUND', detail: '', status: 404 });
    render(<DriverGrid initialDrivers={[driver('VER')]} animate={false} />);
    await waitFor(() => expect(screen.getByTestId('compat')).toHaveTextContent('—'));
  });

  it('reorders cards when sorting by stability', async () => {
    (getStability as Mock).mockImplementation((code: string) =>
      Promise.resolve(stab(code === 'HAM' ? 0.9 : 0.2)),
    );
    (getSetup as Mock).mockResolvedValue(setup(0.5));
    render(<DriverGrid initialDrivers={[driver('VER'), driver('HAM')]} animate={false} />);
    await waitFor(() =>
      expect(screen.getAllByTestId('stability')[0]).not.toHaveTextContent('…'),
    );
    fireEvent.click(screen.getByRole('button', { name: /stability/i }));
    await waitFor(() => {
      const cards = screen.getAllByTestId(/driver-card-/);
      expect(cards[0]).toHaveAttribute('data-testid', 'driver-card-HAM');
    });
  });
});
