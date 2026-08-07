'use client';

import { useEffect, useMemo, useState } from 'react';

import { PageError } from '@/components/ds/PageError';
import { Skeleton } from '@/components/ds/Skeleton';
import type { AnyErrorCode } from '@/lib/api/errors';
import type { DriverSummary, TyreWearResponse } from '@/lib/api/types';
import { getDrivers, getTyres } from '@/lib/api/endpoints';
import { teamColour } from '@/lib/design/teams';

import { DegradationChart } from './DegradationChart';

// Compound display order — the softest degrades fastest; the chips read soft→hard.
const COMPOUND_ORDER = ['SOFT', 'MEDIUM', 'HARD'] as const;

function Tile({
  label,
  value,
  hint,
  accent = false,
}: {
  label: string;
  value: string;
  hint: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-md border border-border bg-card p-4" title={hint}>
      <p className="font-mono text-[8.5px] uppercase tracking-[0.18em] text-text-secondary">
        {label}
      </p>
      <p
        className={`mt-1 font-mono text-[17px] tabular-nums ${
          accent ? 'text-accent' : 'text-text-primary'
        }`}
      >
        {value}
      </p>
    </div>
  );
}

// Screen 4 — Tyre Wear. Every number is measured from clean race stints in the training matrix,
// through the fuel-correction model (each race's fuel slope identified from its pit-stop
// structure, then stripped). The method line under the chart says exactly that; nothing on this
// screen extrapolates beyond the laps behind it.
export function TyreWearScreen() {
  const [wear, setWear] = useState<TyreWearResponse | null>(null);
  const [error, setError] = useState<{ code: AnyErrorCode; detail?: string } | null>(null);
  const [drivers, setDrivers] = useState<DriverSummary[]>([]);
  const [driver, setDriver] = useState<string | null>(null);
  const [compound, setCompound] = useState<string>('MEDIUM');

  useEffect(() => {
    let alive = true;
    void getTyres().then((r) => {
      if (!alive) return;
      if (r.ok) setWear(r.data);
      else setError({ code: r.code, detail: r.detail });
    });
    void getDrivers().then((r) => {
      if (alive && r.ok) setDrivers(r.data);
    });
    return () => {
      alive = false;
    };
  }, []);

  const identities = useMemo(() => new Map(drivers.map((d) => [d.code, d])), [drivers]);
  const byDriver = useMemo(() => wear?.drivers ?? {}, [wear]);
  const codes = useMemo(() => Object.keys(byDriver).sort(), [byDriver]);
  const active = driver ?? codes[0] ?? null;
  const compounds = active ? (byDriver[active] ?? {}) : {};
  const availableCompounds = COMPOUND_ORDER.filter((c) => c in compounds);
  const activeCompound = compound in compounds ? compound : (availableCompounds[0] ?? null);
  const stats = activeCompound ? compounds[activeCompound] : null;
  const colour = teamColour(active ? identities.get(active)?.team : null);

  return (
    <div data-testid="tyre-wear-screen">
      <header className="mb-5">
        <h1 className="font-display text-[34px] font-bold uppercase leading-none tracking-[0.03em] text-text-primary">
          Tyre Wear Analysis
        </h1>
        <p className="mt-2 text-[13px] text-text-secondary">
          Stint degradation measured from clean race laps, fuel-corrected per race.
        </p>
      </header>

      {error !== null ? (
        <PageError code={error.code} detail={error.detail} />
      ) : wear === null ? (
        <Skeleton className="h-64 w-full" />
      ) : codes.length === 0 ? (
        <p className="font-mono text-[11px] text-text-secondary" data-testid="no-tyre-data">
          No tyre-wear data — build the driver profiles artefact first.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-start justify-between gap-3 rounded-md border border-border bg-card p-4">
            <div>
              <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                Driver
              </p>
              <div className="flex flex-wrap gap-1.5" data-testid="driver-chips">
                {codes.map((code) => {
                  const selected = code === active;
                  return (
                    <button
                      key={code}
                      type="button"
                      onClick={() => setDriver(code)}
                      data-testid={`driver-chip-${code}`}
                      data-selected={selected || undefined}
                      className={`rounded border px-2.5 py-1 font-mono text-[11px] transition-colors focus:outline-none focus:ring-1 focus:ring-accent ${
                        selected
                          ? 'border-accent bg-accent/10 text-text-primary'
                          : 'border-border text-text-secondary hover:border-text-secondary hover:text-text-primary'
                      }`}
                    >
                      {code}
                    </button>
                  );
                })}
              </div>
            </div>
            <div>
              <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
                Compound
              </p>
              <div className="flex gap-1.5" data-testid="compound-chips">
                {availableCompounds.map((c) => {
                  const selected = c === activeCompound;
                  return (
                    <button
                      key={c}
                      type="button"
                      onClick={() => setCompound(c)}
                      data-testid={`compound-chip-${c}`}
                      data-selected={selected || undefined}
                      className={`rounded border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.1em] transition-colors focus:outline-none focus:ring-1 focus:ring-accent ${
                        selected
                          ? 'border-confidence-medium bg-confidence-medium/10 text-text-primary'
                          : 'border-border text-text-secondary hover:border-text-secondary hover:text-text-primary'
                      }`}
                    >
                      {c}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {stats ? (
            <>
              <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
                <Tile
                  label="Median stint"
                  value={`${stats.median_stint_len} laps`}
                  hint={`${stats.n_stints} qualifying stints, ${stats.n_laps} clean laps; longest ${stats.max_stint_len} laps`}
                />
                <Tile
                  label="Degradation rate"
                  value={`+${stats.deg_slope_s_per_lap.toFixed(3)}s/lap`}
                  hint="Median within-stint slope of fuel-corrected lap time vs tyre age"
                  accent
                />
                <Tile
                  label="Total delta"
                  value={`+${stats.total_delta_s.toFixed(2)}s`}
                  hint="Degradation rate × (median stint length − 1) — pace lost over a typical stint"
                />
                <Tile
                  label="Cliff lap (est.)"
                  value={stats.cliff_age != null ? `Lap ${stats.cliff_age}` : '—'}
                  hint={
                    stats.cliff_age != null
                      ? 'First tyre age where the curve accelerates clearly past its linear trend'
                      : 'No acceleration past the linear trend detected in this data'
                  }
                />
              </div>

              <section className="mt-4 rounded-md border border-border bg-card p-4">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
                    Lap-time degradation — fuel-corrected
                  </h2>
                  <span className="rounded border border-border px-2 py-1 font-mono text-[9px] uppercase tracking-[0.16em] text-text-primary">
                    {active} / {activeCompound}
                  </span>
                </div>
                <DegradationChart stats={stats} colour={colour} label={`${active} ${activeCompound}`} />
                <p className="mt-3 max-w-4xl font-mono text-[10px] leading-relaxed text-text-secondary">
                  Median lap-time delta vs each stint&apos;s own early-stint baseline, aggregated over{' '}
                  {stats.n_stints} race stints; the band is the interquartile range. Fuel effect (
                  {stats.fuel_slope_s_per_lap != null
                    ? `${stats.fuel_slope_s_per_lap.toFixed(3)}s per race lap, `
                    : ''}
                  identified per race from its pit-stop structure) is stripped before aggregation, so
                  the curve isolates the tyre.
                  {stats.f27_median != null || stats.f28_median != null ? (
                    <>
                      {' '}
                      Behavioural adaptation over the same stints:
                      {stats.f27_median != null
                        ? ` throttle drift ${stats.f27_median >= 0 ? '+' : ''}${stats.f27_median.toFixed(3)}/lap`
                        : ''}
                      {stats.f27_median != null && stats.f28_median != null ? ' ·' : ''}
                      {stats.f28_median != null
                        ? ` brake drift ${stats.f28_median >= 0 ? '+' : ''}${stats.f28_median.toFixed(3)}/lap`
                        : ''}{' '}
                      (F27/F28).
                    </>
                  ) : null}
                </p>
              </section>
            </>
          ) : (
            <p className="mt-4 font-mono text-[11px] text-text-secondary" data-testid="no-compound-data">
              No qualifying stints for this driver on this compound.
            </p>
          )}
        </>
      )}
    </div>
  );
}
