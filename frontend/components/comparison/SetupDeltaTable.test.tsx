import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ApiResult } from '@/lib/api/errors';
import type { SetupResponse } from '@/lib/api/types';

import { SetupDeltaTable } from './SetupDeltaTable';

type Param = { name: string; value: number | string | null };

const setup = (params: Param[]): ApiResult<SetupResponse> =>
  ({
    ok: true,
    data: {
      driver: 'X',
      circuit: 'Bahrain',
      parameters: params.map((p) => ({ ...p, kind: 'NUM', confidence: 0.8, rating: 'HIGH' })),
      approximation_disclaimer: '',
      model_version: 'v1',
    },
    status: 200,
    partial: false,
    flags: [],
  }) as unknown as ApiResult<SetupResponse>;

const notFound = { ok: false, code: 'NOT_FOUND', detail: '', status: 404 } as ApiResult<SetupResponse>;

describe('SetupDeltaTable (master §10.5)', () => {
  it('diffs shared parameters by name, sorted by |Δ|', () => {
    render(
      <SetupDeltaTable
        driver1="VER"
        driver2="HAM"
        setup1={setup([
          { name: 'front_wing', value: 8 },
          { name: 'brake_bias', value: 56 },
        ])}
        setup2={setup([
          { name: 'front_wing', value: 6 },
          { name: 'brake_bias', value: 55 },
        ])}
      />,
    );
    const rows = screen.getAllByTestId('setup-delta-row');
    expect(rows[0]).toHaveAttribute('data-param', 'front_wing'); // |Δ|=2 > |Δ|=1
    expect(screen.getByTestId('setup-delta-table')).toBeInTheDocument();
  });

  it('renders an honest EmptyState when a driver has no setup for the circuit (404)', () => {
    render(
      <SetupDeltaTable driver1="VER" driver2="HAM" setup1={setup([{ name: 'front_wing', value: 8 }])} setup2={notFound} />,
    );
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });
});
