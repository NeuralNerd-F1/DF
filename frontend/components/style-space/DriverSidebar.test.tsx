import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ getFingerprint: vi.fn() }));

import { getFingerprint } from '@/lib/api/endpoints';

import type { StyleNode } from '@/lib/style-space/styleSpace';

import { DriverSidebar } from './DriverSidebar';

const node: StyleNode = {
  code: 'VER',
  coords2d: [0, 0],
  coords5d: [1, 0, 0],
  team: 'Red Bull',
  radar: { throttle: 0.6 },
};

beforeEach(() => (getFingerprint as Mock).mockResolvedValue({ ok: false, code: 'NOT_FOUND', detail: '', status: 404 }));
afterEach(() => vi.clearAllMocks());

describe('DriverSidebar (master §10.6)', () => {
  it('renders code, team and the fingerprint radar', () => {
    render(<DriverSidebar node={node} onClose={() => {}} />);
    expect(screen.getByTestId('sidebar-code')).toHaveTextContent('VER');
    expect(screen.getByText('Red Bull')).toBeInTheDocument();
    expect(screen.getByTestId('fingerprint-radar')).toBeInTheDocument();
  });

  it('shows nl_description when the fetch succeeds (degrades silently otherwise)', async () => {
    (getFingerprint as Mock).mockResolvedValue({
      ok: true,
      data: { nl_description: 'Aggressive late braker.' },
      status: 200,
      partial: false,
      flags: [],
    });
    render(<DriverSidebar node={node} onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText('Aggressive late braker.')).toBeInTheDocument());
  });

  it('calls onClose from the close button', () => {
    const onClose = vi.fn();
    render(<DriverSidebar node={node} onClose={onClose} />);
    fireEvent.click(screen.getByTestId('sidebar-close'));
    expect(onClose).toHaveBeenCalled();
  });
});
