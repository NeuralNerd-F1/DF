'use client';

import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import type { ApiResult } from '@/lib/api/errors';
import type { SetupParameterSchema, SetupResponse } from '@/lib/api/types';

// Setup delta (master §10.5): which setup parameters differ most between the two drivers' optimal
// setups for the selected circuit. Fully real — two GET /drivers/{id}/setup/{circuit} calls diffed
// by parameter name, sorted by |Δ|. Reuses the F11 SetupParameterSchema.
type Row = {
  name: string;
  v1: number | string | null;
  v2: number | string | null;
  delta: number | null; // numeric params only
};

function valueOf(p: SetupParameterSchema): number | string | null {
  const v = p.value;
  if (typeof v === 'number' || typeof v === 'string') return v;
  return null;
}

function buildRows(a: SetupResponse, b: SetupResponse): Row[] {
  const byName = new Map(b.parameters.map((p) => [p.name, p]));
  const rows: Row[] = [];
  for (const p1 of a.parameters) {
    const p2 = byName.get(p1.name);
    if (!p2) continue;
    const v1 = valueOf(p1);
    const v2 = valueOf(p2);
    const delta = typeof v1 === 'number' && typeof v2 === 'number' ? v1 - v2 : null;
    rows.push({ name: p1.name, v1, v2, delta });
  }
  return rows.sort((x, y) => Math.abs(y.delta ?? 0) - Math.abs(x.delta ?? 0));
}

function cell(v: number | string | null): string {
  if (v == null) return '—';
  return typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : v;
}

export function SetupDeltaTable({
  setup1,
  setup2,
  driver1,
  driver2,
}: {
  setup1: ApiResult<SetupResponse> | null;
  setup2: ApiResult<SetupResponse> | null;
  driver1: string;
  driver2: string;
}) {
  if (!setup1 || !setup2) return <Skeleton className="h-24" label="Loading setups" />;

  // A 404 for either driver is honest absence, not an error (no setup for this circuit).
  if ((!setup1.ok && setup1.code === 'NOT_FOUND') || (!setup2.ok && setup2.code === 'NOT_FOUND')) {
    return <EmptyState message="Setup tendencies not available for one driver at this circuit." className="py-3" />;
  }
  if (!setup1.ok) return <ErrorCard code={setup1.code} detail={setup1.detail} />;
  if (!setup2.ok) return <ErrorCard code={setup2.code} detail={setup2.detail} />;

  const rows = buildRows(setup1.data, setup2.data);
  if (rows.length === 0) {
    return <EmptyState message="No shared setup parameters to compare." className="py-3" />;
  }

  return (
    <table data-testid="setup-delta-table" className="w-full font-mono text-[11px]">
      <thead>
        <tr className="text-text-secondary">
          <th className="py-1 text-left">Parameter</th>
          <th className="py-1 text-right">{driver1}</th>
          <th className="py-1 text-right">{driver2}</th>
          <th className="py-1 text-right">Δ</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.name} data-testid="setup-delta-row" data-param={r.name} className="border-t border-border">
            <td className="py-1 capitalize text-text-primary">{r.name.replace(/_/g, ' ')}</td>
            <td className="py-1 text-right text-text-primary">{cell(r.v1)}</td>
            <td className="py-1 text-right text-text-primary">{cell(r.v2)}</td>
            <td className="py-1 text-right text-text-secondary" data-testid="setup-delta-value">
              {r.delta == null ? '—' : `${r.delta >= 0 ? '+' : ''}${cell(r.delta)}`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
