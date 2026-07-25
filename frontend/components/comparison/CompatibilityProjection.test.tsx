import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ApiResult } from '@/lib/api/errors';
import type { SetupResponse } from '@/lib/api/types';

import { CompatibilityProjection } from './CompatibilityProjection';

const setup = (conf: number): ApiResult<SetupResponse> =>
  ({
    ok: true,
    data: { parameters: [{ confidence: conf }, { confidence: conf }] },
    status: 200,
    partial: false,
    flags: [],
  }) as unknown as ApiResult<SetupResponse>;

const notFound = { ok: false, code: 'NOT_FOUND', detail: '', status: 404 } as ApiResult<SetupResponse>;
const outage = { ok: false, code: 'DB_UNAVAILABLE', detail: 'down', status: 503 } as ApiResult<SetupResponse>;

describe('CompatibilityProjection (master §10.5, div #62/#47)', () => {
  it('shows the delta of two setup-reliability proxies', () => {
    render(<CompatibilityProjection driver1="VER" driver2="HAM" setup1={setup(0.8)} setup2={setup(0.6)} />);
    expect(screen.getByTestId('gauge-value')).toHaveTextContent('0.2');
  });

  it('renders an honest "—" when a driver setup is absent (404)', () => {
    render(<CompatibilityProjection driver1="VER" driver2="HAM" setup1={setup(0.8)} setup2={notFound} />);
    expect(screen.getByTestId('gauge-value')).toHaveTextContent('—');
    const markers = screen.getAllByTestId('compat-driver');
    expect(markers.some((m) => m.textContent?.includes('—'))).toBe(true);
  });

  it('distinguishes a real outage (503) with a code marker, not an absence', () => {
    render(<CompatibilityProjection driver1="VER" driver2="HAM" setup1={setup(0.8)} setup2={outage} />);
    const marker = screen.getAllByTestId('compat-driver').find((m) => m.querySelector('[data-metric-error]'));
    expect(marker).toBeDefined();
    expect(marker).toHaveTextContent('DB_UNAVAILABLE');
  });
});
