'use client';

import { useEffect, useState } from 'react';

import { DeltaRadar } from '@/components/charts/DeltaRadar';
import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import { Button } from '@/components/ds/Button';
import { Card } from '@/components/ds/Card';
import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { getCompare, getSetup } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type { CompareResponse, DriverSummary, SetupResponse } from '@/lib/api/types';
import { CIRCUITS, DEFAULT_CIRCUIT } from '@/lib/design/circuits';

import { CompatibilityProjection } from './CompatibilityProjection';
import { DistinctiveFeatures } from './DistinctiveFeatures';
import { SetupDeltaTable } from './SetupDeltaTable';

// Screen 4 — Comparison (master §10.5). Two drivers side by side + a signed delta radar, the
// most-distinctive domains, a compatibility projection (proxy, div #62) and a setup delta. Swap
// flips the delta sign (driver_1/driver_2 swap in /compare). Animation off in tests.
const fieldClass =
  'rounded border border-border bg-background px-2 py-1 font-mono text-xs text-text-primary';

export function ComparisonScreen({
  initialDrivers,
  animate = true,
}: {
  initialDrivers: DriverSummary[];
  animate?: boolean;
}) {
  const codes = initialDrivers.map((d) => d.code);
  const [d1, setD1] = useState<string>(codes[0] ?? '');
  const [d2, setD2] = useState<string>(codes[1] ?? codes[0] ?? '');
  const [circuit, setCircuit] = useState<string>(DEFAULT_CIRCUIT);

  const [compare, setCompare] = useState<ApiResult<CompareResponse> | null>(null);
  const [setup1, setSetup1] = useState<ApiResult<SetupResponse> | null>(null);
  const [setup2, setSetup2] = useState<ApiResult<SetupResponse> | null>(null);

  useEffect(() => {
    if (!d1 || !d2) return;
    let alive = true;
    setCompare(null);
    void getCompare(d1, d2).then((r) => {
      if (alive) setCompare(r);
    });
    return () => {
      alive = false;
    };
  }, [d1, d2]);

  useEffect(() => {
    if (!d1 || !d2) return;
    let alive = true;
    setSetup1(null);
    setSetup2(null);
    void getSetup(d1, circuit).then((r) => alive && setSetup1(r));
    void getSetup(d2, circuit).then((r) => alive && setSetup2(r));
    return () => {
      alive = false;
    };
  }, [d1, d2, circuit]);

  function swap() {
    setD1(d2);
    setD2(d1);
  }

  return (
    <main className="mx-auto max-w-5xl p-6" data-testid="comparison-screen">
      <h1 className="mb-4 font-mono text-lg uppercase tracking-wide text-accent">Comparison</h1>

      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] text-text-secondary">Driver 1</span>
          <select data-testid="driver1-select" aria-label="Driver 1" value={d1} onChange={(e) => setD1(e.target.value)} className={fieldClass}>
            {codes.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <Button onClick={swap} data-testid="swap-button">
          ⇄ Swap
        </Button>
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] text-text-secondary">Driver 2</span>
          <select data-testid="driver2-select" aria-label="Driver 2" value={d2} onChange={(e) => setD2(e.target.value)} className={fieldClass}>
            {codes.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <label className="ml-auto flex flex-col gap-1">
          <span className="font-mono text-[10px] text-text-secondary">Circuit</span>
          <select data-testid="compare-circuit" aria-label="Circuit" value={circuit} onChange={(e) => setCircuit(e.target.value)} className={fieldClass}>
            {CIRCUITS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
      </div>

      {compare == null ? (
        <Skeleton className="h-64" label="Loading comparison" />
      ) : !compare.ok ? (
        <ErrorCard code={compare.code} detail={compare.detail} />
      ) : (
        <div className="grid gap-6 md:grid-cols-3">
          <Card className="flex flex-col items-center p-4">
            <h2 className="mb-2 font-mono text-xs text-text-secondary" data-testid="radar1-label">{d1}</h2>
            <FingerprintRadar values={compare.data.driver_1_radar} animate={animate} ariaLabel={`${d1} fingerprint`} />
          </Card>
          <Card className="flex flex-col items-center p-4">
            <h2 className="mb-2 font-mono text-xs text-text-secondary">Δ (signed)</h2>
            <DeltaRadar delta={compare.data.delta_shap ?? {}} animate={animate} />
          </Card>
          <Card className="flex flex-col items-center p-4">
            <h2 className="mb-2 font-mono text-xs text-text-secondary" data-testid="radar2-label">{d2}</h2>
            <FingerprintRadar values={compare.data.driver_2_radar} animate={animate} ariaLabel={`${d2} fingerprint`} />
          </Card>
        </div>
      )}

      {compare?.ok ? (
        <div className="mt-6 grid gap-6 md:grid-cols-3">
          <Card className="p-4">
            <h2 className="mb-2 font-mono text-xs uppercase text-text-secondary">Most distinctive (domains)</h2>
            <DistinctiveFeatures compare={compare.data} />
          </Card>
          <Card className="p-4">
            <h2 className="mb-2 font-mono text-xs uppercase text-text-secondary">Compatibility</h2>
            <CompatibilityProjection setup1={setup1} setup2={setup2} driver1={d1} driver2={d2} />
          </Card>
          <Card className="p-4">
            <h2 className="mb-2 font-mono text-xs uppercase text-text-secondary">Setup delta</h2>
            <SetupDeltaTable setup1={setup1} setup2={setup2} driver1={d1} driver2={d2} />
          </Card>
        </div>
      ) : null}

      {codes.length < 2 ? <EmptyState message="Need at least two drivers to compare." className="mt-6" /> : null}
    </main>
  );
}
