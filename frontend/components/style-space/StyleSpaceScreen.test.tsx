import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

// Stub next/dynamic so the WebGL-branch test does not load real `three` (the 2D path is the tested
// one; the scene has its own smoke test).
vi.mock('next/dynamic', () => ({
  default: () =>
    function SceneStub() {
      return <div data-testid="scene-stub" />;
    },
}));
vi.mock('@/lib/api/endpoints', () => ({ getFingerprint: vi.fn(), postUnknownFingerprint: vi.fn() }));

import { getFingerprint, postUnknownFingerprint } from '@/lib/api/endpoints';

import type { DriverSummary, StyleSpaceResponse } from '@/lib/api/types';

import { StyleSpaceScreen } from './StyleSpaceScreen';

const STYLE = {
  drivers: [
    { id: 'VER', code: 'VER', coords_2d: [0, 0], coords_5d: [1, 0, 0], season: null },
    { id: 'PER', code: 'PER', coords_2d: [1, 1], coords_5d: [0.9, 0.1, 0], season: null },
  ],
  unknown_drivers: [],
} as unknown as StyleSpaceResponse;

const DRIVERS = [
  { id: 'VER', code: 'VER', team: 'Red Bull', radar_values: { throttle: 0.6 } },
  { id: 'PER', code: 'PER', team: 'Red Bull', radar_values: { throttle: 0.4 } },
] as unknown as DriverSummary[];

const EMPTY = { drivers: [], unknown_drivers: [] } as unknown as StyleSpaceResponse;

beforeEach(() => {
  (getFingerprint as Mock).mockResolvedValue({ ok: false, code: 'NOT_FOUND', detail: '', status: 404 });
  (postUnknownFingerprint as Mock).mockResolvedValue({ ok: false, code: 'INVALID_INPUT', detail: '', status: 400 });
});
afterEach(() => vi.clearAllMocks());

describe('StyleSpaceScreen (master §10.6)', () => {
  it('renders the D3 2D fallback with NO error in jsdom (silent, WebGL absent — div #68)', () => {
    render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} />);
    expect(screen.getByTestId('style-space-screen')).toHaveAttribute('data-mode', '2d');
    expect(screen.getByTestId('style-space-2d')).toBeInTheDocument();
    expect(screen.queryByTestId('error-card')).toBeNull();
  });

  it('selects the WebGL branch when it is available (forced)', () => {
    render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} forceMode="webgl" />);
    expect(screen.getByTestId('style-space-screen')).toHaveAttribute('data-mode', 'webgl');
    expect(screen.getByTestId('scene-stub')).toBeInTheDocument();
  });

  it('renders an EmptyState for an empty style space', () => {
    render(<StyleSpaceScreen initialStyle={EMPTY} initialDrivers={[] as unknown as DriverSummary[]} />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });

  it('shows a tooltip on hover and a sidebar on click', () => {
    render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} />);
    const ver = document.querySelector('[data-driver="VER"]')!;
    fireEvent.mouseEnter(ver);
    expect(screen.getByTestId('driver-tooltip')).toBeInTheDocument();
    expect(screen.getByTestId('tooltip-code')).toHaveTextContent('VER');
    fireEvent.click(ver);
    expect(screen.getByTestId('driver-sidebar')).toBeInTheDocument();
  });

  it('renders the season filter as a deferred/disabled control (div #64)', () => {
    render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} />);
    const btns = screen.getAllByTestId('season-btn');
    expect(btns).toHaveLength(3);
    expect(btns.every((b) => (b as HTMLButtonElement).disabled)).toBe(true);
  });
});
