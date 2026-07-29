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

  // The legend doubles as a driver picker, so it must drive the SAME selection/hover state as the
  // scatter — a spy on the prop only proves the callback fired, not that the screen is wired to it.
  it('a legend row selects and hovers the same driver as the scatter (shared state)', () => {
    render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} />);
    expect(screen.queryByTestId('driver-sidebar')).toBeNull();

    fireEvent.mouseEnter(screen.getByTestId('legend-row-PER'));
    expect(screen.getByTestId('tooltip-code')).toHaveTextContent('PER');

    fireEvent.click(screen.getByTestId('legend-row-PER'));
    expect(screen.getByTestId('driver-sidebar')).toHaveTextContent('PER');
    expect(screen.getByTestId('legend-row-PER')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('legend-row-VER')).toHaveAttribute('aria-pressed', 'false');
  });

  // The caption explains what the dots are; it must not claim capabilities the active mode lacks
  // (2D renders no unknown sphere and has no orbit) nor name an engine /stylespace never reports.
  it('captions the scatter honestly per mode, and never names UMAP as the confirmed engine', () => {
    const { unmount } = render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} />);
    const flat = screen.getByTestId('style-space-about');
    expect(flat).toHaveTextContent('56-number summary');
    expect(flat).toHaveTextContent('UMAP, or its PCA fallback'); // hedged — engine is not exposed
    expect(flat).toHaveTextContent('flat 2D projection');
    expect(flat).not.toHaveTextContent('drifting white sphere'); // 2D draws no unknown marker
    expect(flat).not.toHaveTextContent('scroll to zoom');
    unmount();

    render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} forceMode="webgl" />);
    const scene = screen.getByTestId('style-space-about');
    expect(scene).toHaveTextContent('plots 3 of them');
    expect(scene).toHaveTextContent('drifting white sphere');
  });

  it('renders the season filter as a deferred/disabled control (div #64)', () => {
    render(<StyleSpaceScreen initialStyle={STYLE} initialDrivers={DRIVERS} />);
    const btns = screen.getAllByTestId('season-btn');
    expect(btns).toHaveLength(3);
    expect(btns.every((b) => (b as HTMLButtonElement).disabled)).toBe(true);
  });
});
