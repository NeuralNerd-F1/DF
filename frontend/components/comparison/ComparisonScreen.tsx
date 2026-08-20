'use client';

import { useEffect, useMemo, useState } from 'react';

import { DeltaRadar } from '@/components/charts/DeltaRadar';
import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { getCompare, getFingerprint, getSetup } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type {
  CompareResponse,
  DriverSummary,
  FingerprintResponse,
  SetupResponse,
} from '@/lib/api/types';
import { featureShapDelta, pairAxisMaxima, scaleByMaxima } from '@/lib/compare/compare';
import { CIRCUITS, DEFAULT_CIRCUIT } from '@/lib/design/circuits';
import { teamColour } from '@/lib/design/teams';

import { CompatibilityProjection } from './CompatibilityProjection';
import { DistinctiveFeatures } from './DistinctiveFeatures';
import { SetupDeltaTable } from './SetupDeltaTable';

// Screen 4 — Comparison (master §10.5), rebuilt in the Screens-1-4 design language. Two team-
// coloured radars + a signed delta radar on ONE shared per-axis scale (each axis divided by the
// PAIR's max — feeding the raw served |SHAP| magnitudes into the 0-1 radar clamps most axes to the
// rim and makes every driver look identical), the top-5 per-FEATURE signed-SHAP delta (from the two
// served fingerprints), the compatibility proxy (div #62 — labelled, not circuit-aware), and the
// setup delta table. Swap exchanges the drivers (the /compare order flips, so the delta sign flips
// with it). Animation off in tests.
const chip = (selected: boolean): string =>
  `rounded border px-2.5 py-1 font-mono text-[11px] transition-colors focus:outline-none focus:ring-1 focus:ring-accent ${
    selected
      ? 'border-accent-deep bg-accent-fill text-text-primary'
      : 'border-border text-text-secondary hover:border-text-secondary hover:text-text-primary'
  }`;

export function ComparisonScreen({
  initialDrivers,
  animate = true,
}: {
  initialDrivers: DriverSummary[];
  animate?: boolean;
}) {
  const codes = useMemo(() => initialDrivers.map((d) => d.code), [initialDrivers]);
  const identities = useMemo(
    () => new Map(initialDrivers.map((d) => [d.code, d])),
    [initialDrivers],
  );
  const [d1, setD1] = useState<string>(codes[0] ?? '');
  const [d2, setD2] = useState<string>(codes[1] ?? codes[0] ?? '');
  const [circuit, setCircuit] = useState<string>(DEFAULT_CIRCUIT);

  const [compare, setCompare] = useState<ApiResult<CompareResponse> | null>(null);
  const [fp1, setFp1] = useState<ApiResult<FingerprintResponse> | null>(null);
  const [fp2, setFp2] = useState<ApiResult<FingerprintResponse> | null>(null);
  const [setup1, setSetup1] = useState<ApiResult<SetupResponse> | null>(null);
  const [setup2, setSetup2] = useState<ApiResult<SetupResponse> | null>(null);

  useEffect(() => {
    if (!d1 || !d2) return;
    let alive = true;
    setCompare(null);
    setFp1(null);
    setFp2(null);
    void getCompare(d1, d2).then((r) => alive && setCompare(r));
    // The per-feature signed SHAP comes from the two served fingerprints; a failure here only
    // degrades the distinctive-features panel to the per-domain delta (handled there, labelled).
    void getFingerprint(d1).then((r) => alive && setFp1(r));
    void getFingerprint(d2).then((r) => alive && setFp2(r));
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

  const colour1 = teamColour(identities.get(d1)?.team);
  const colour2 = teamColour(identities.get(d2)?.team);

  // One shared 0-1 scale per axis for both radars AND the delta (comparability requires it).
  const scaled = useMemo(() => {
    if (compare == null || !compare.ok) return null;
    const maxima = pairAxisMaxima(compare.data.driver_1_radar, compare.data.driver_2_radar);
    const delta = Object.fromEntries(
      Object.entries(compare.data.delta_shap ?? {}).map(([k, v]) => [
        k,
        maxima[k] != null && maxima[k] > 0 ? v / maxima[k] : 0,
      ]),
    );
    return {
      r1: scaleByMaxima(compare.data.driver_1_radar, maxima),
      r2: scaleByMaxima(compare.data.driver_2_radar, maxima),
      delta,
    };
  }, [compare]);

  const featureDelta = useMemo(() => {
    if (fp1?.ok && fp2?.ok) {
      const s1 = fp1.data.shap_signed ?? {};
      const s2 = fp2.data.shap_signed ?? {};
      if (Object.keys(s1).length > 0 && Object.keys(s2).length > 0) {
        return featureShapDelta(s1, s2);
      }
    }
    return null;
  }, [fp1, fp2]);

  if (codes.length < 2) {
    return (
      <div data-testid="comparison-screen">
        <Header />
        <EmptyState message="Need at least two drivers to compare." className="mt-6" />
      </div>
    );
  }

  return (
    <div data-testid="comparison-screen">
      <Header />

      {/* Selection — two driver slots + swap, circuit for the setup-based panels. */}
      <div className="mb-4 flex flex-wrap items-start justify-between gap-4 rounded-md border border-border bg-card p-4">
        <DriverPicker slot={1} codes={codes} value={d1} onPick={setD1} />
        <div className="self-center">
          <button
            type="button"
            onClick={swap}
            data-testid="swap-button"
            className="rounded border border-border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.1em] text-accent transition-colors hover:border-accent"
          >
            ⇄ Swap
          </button>
        </div>
        <DriverPicker slot={2} codes={codes} value={d2} onPick={setD2} />
        <div>
          <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
            Circuit (setup panels)
          </p>
          <div className="flex max-w-xs flex-wrap gap-1.5" data-testid="circuit-chips">
            {CIRCUITS.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => setCircuit(c)}
                data-testid={`circuit-chip-${c}`}
                data-selected={c === circuit || undefined}
                className={chip(c === circuit)}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
      </div>

      {compare == null ? (
        <Skeleton className="h-64" label="Loading comparison" />
      ) : !compare.ok ? (
        compare.status === 404 || compare.code === 'NOT_FOUND' ? (
          <EmptyState message="No fingerprint for one of these drivers." />
        ) : (
          <ErrorCard code={compare.code} detail={compare.detail} />
        )
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <RadarCard
              code={d1}
              identity={identities.get(d1)}
              values={scaled?.r1 ?? {}}
              colour={colour1}
              animate={animate}
              testid="radar1-label"
            />
            <section className="flex flex-col items-center rounded-md border border-border bg-card p-4">
              <h2 className="mb-1 self-start font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                Signed delta
              </h2>
              <DeltaRadar delta={scaled?.delta ?? {}} animate={animate} />
              <p className="mt-1 self-start font-mono text-[9px] text-text-secondary" data-testid="delta-caption">
                red = {d1} higher · blue = {d2} higher
              </p>
            </section>
            <RadarCard
              code={d2}
              identity={identities.get(d2)}
              values={scaled?.r2 ?? {}}
              colour={colour2}
              animate={animate}
              testid="radar2-label"
            />
          </div>
          <p className="mt-2 font-mono text-[9px] text-text-secondary" data-testid="radar-scale-note">
            Axes show mean |SHAP| attribution per domain, each axis scaled to this pair&apos;s
            maximum. The two radars and the delta all share one scale.
          </p>

          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <section className="rounded-md border border-border bg-card p-4">
              <h2 className="mb-3 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                Most distinctive features
              </h2>
              <DistinctiveFeatures
                featureDelta={featureDelta}
                domainDelta={compare.data.delta_shap ?? {}}
                driver1={d1}
                driver2={d2}
              />
            </section>
            <section className="rounded-md border border-border bg-card p-4">
              <h2 className="mb-3 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                Compatibility at {circuit}
              </h2>
              <CompatibilityProjection setup1={setup1} setup2={setup2} driver1={d1} driver2={d2} />
            </section>
            <section className="rounded-md border border-border bg-card p-4">
              <h2 className="mb-3 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                Setup delta at {circuit}
              </h2>
              <SetupDeltaTable setup1={setup1} setup2={setup2} driver1={d1} driver2={d2} />
            </section>
          </div>
        </>
      )}
    </div>
  );
}

function Header() {
  return (
    <header className="mb-5">
      <h1 className="font-display text-[34px] font-bold uppercase leading-none tracking-[0.03em] text-text-primary">
        Comparison
      </h1>
      <p className="mt-2 text-[13px] text-text-secondary">
        Two behavioural fingerprints side by side. Where the model tells them apart, and what
        that implies for the car.
      </p>
    </header>
  );
}

function DriverPicker({
  slot,
  codes,
  value,
  onPick,
}: {
  slot: 1 | 2;
  codes: string[];
  value: string;
  onPick: (code: string) => void;
}) {
  return (
    <div>
      <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
        Driver {slot}
      </p>
      <div className="flex max-w-xs flex-wrap gap-1.5" data-testid={`driver${slot}-chips`}>
        {codes.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => onPick(c)}
            data-testid={`driver${slot}-chip-${c}`}
            data-selected={c === value || undefined}
            className={chip(c === value)}
          >
            {c}
          </button>
        ))}
      </div>
    </div>
  );
}

function RadarCard({
  code,
  identity,
  values,
  colour,
  animate,
  testid,
}: {
  code: string;
  identity: DriverSummary | undefined;
  values: Record<string, number>;
  colour: string;
  animate: boolean;
  testid: string;
}) {
  return (
    <section className="flex flex-col items-center rounded-md border border-border bg-card p-4">
      <div className="mb-1 flex w-full items-baseline justify-between">
        <h2 className="font-mono text-[13px] uppercase tracking-[0.14em] text-text-primary" data-testid={testid}>
          {code}
        </h2>
        <span className="font-mono text-[8.5px] uppercase tracking-[0.14em] text-text-secondary">
          {identity?.team ?? ''}
        </span>
      </div>
      <FingerprintRadar values={values} animate={animate} colour={colour} ariaLabel={`${code} fingerprint`} />
    </section>
  );
}
