import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { SystemStatus } from './SystemStatus';

const getHealth = vi.hoisted(() => vi.fn());
vi.mock('@/lib/api/endpoints', () => ({ getHealth }));

const healthy = (models: Record<string, boolean>) => ({
  ok: true as const,
  data: { status: 'healthy', models_loaded: models, db_connected: true, version: '0.1.0' },
});

describe('SystemStatus — claims resolved from the real /health, never assumed', () => {
  beforeEach(() => getHealth.mockReset());

  it('every model loaded → MODEL ACTIVE', async () => {
    getHealth.mockResolvedValue(healthy({ classifier: true, stylespace: true }));
    render(<SystemStatus />);
    await waitFor(() =>
      expect(screen.getByTestId('system-status')).toHaveAttribute('data-status', 'active'),
    );
    expect(screen.getByTestId('system-status')).toHaveTextContent(/model active/i);
  });

  it('some models missing → the honest count, not a green light', async () => {
    getHealth.mockResolvedValue(healthy({ classifier: true, stylespace: false, deg: false }));
    render(<SystemStatus />);
    await waitFor(() =>
      expect(screen.getByTestId('system-status')).toHaveAttribute('data-status', 'partial'),
    );
    expect(screen.getByTestId('system-status')).toHaveTextContent('1/3 models');
  });

  it('API unreachable → API OFFLINE', async () => {
    getHealth.mockResolvedValue({ ok: false, code: 'API_ERROR', detail: 'down' });
    render(<SystemStatus />);
    await waitFor(() =>
      expect(screen.getByTestId('system-status')).toHaveAttribute('data-status', 'offline'),
    );
    expect(screen.getByTestId('system-status')).toHaveTextContent(/api offline/i);
  });

  it('renders the probing state before the response lands (hydration-safe initial)', () => {
    getHealth.mockReturnValue(new Promise(() => {}));
    render(<SystemStatus />);
    expect(screen.getByTestId('system-status')).toHaveAttribute('data-status', 'probing');
  });
});
