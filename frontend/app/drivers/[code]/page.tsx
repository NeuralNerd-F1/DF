import Link from 'next/link';

import { FactsCard } from '@/components/driver-profile/FactsCard';
import { ProfileHeader } from '@/components/driver-profile/ProfileHeader';
import { SetupList } from '@/components/driver-profile/SetupList';
import { SignalWeightsCard } from '@/components/driver-profile/SignalWeightsCard';
import { StyleMapCard } from '@/components/driver-profile/StyleMapCard';
import { StyleProfileCard } from '@/components/driver-profile/StyleProfileCard';
import { PageError } from '@/components/ds/PageError';
import { getDrivers, getFingerprint, getSetup, getStyleSpace } from '@/lib/api/endpoints';
import { DEFAULT_CIRCUIT } from '@/lib/design/circuits';
import { gridRelativeRadar } from '@/lib/design/radar';
import { teamColour } from '@/lib/design/teams';

export const dynamic = 'force-dynamic';

// Screen 2 — Driver Profile. Four independent data sources, each degrading on its own: the
// fingerprint (radar + signed SHAP + NL summary) is load-bearing and errors the page; the drivers
// list (identity, facts, grid-relative radar scale), setup and style space each degrade to an
// honest note without taking the rest of the screen down.
export default async function DriverProfilePage({ params }: { params: { code: string } }) {
  const code = params.code;
  const [fp, driversRes, setupRes, styleRes] = await Promise.all([
    getFingerprint(code),
    getDrivers(),
    getSetup(code, DEFAULT_CIRCUIT),
    getStyleSpace(),
  ]);

  if (!fp.ok) {
    return (
      <main className="mx-auto max-w-7xl p-6">
        <Link
          href="/"
          className="font-mono text-[11px] uppercase tracking-[0.15em] text-accent hover:underline"
        >
          ← Back to overview
        </Link>
        <div className="mt-4">
          <PageError code={fp.code} detail={fp.detail} />
        </div>
      </main>
    );
  }

  const drivers = driversRes.ok ? driversRes.data : [];
  const driver = drivers.find((d) => d.code === code) ?? {
    id: code,
    code,
    team: null,
    full_name: null,
    radar_values: fp.data.shap_radar ?? {},
    signature: [],
    profile: null,
  };
  const colour = teamColour(driver.team);
  // Grid-relative scale needs the whole grid; with only this driver available the radar scales
  // against itself (its strongest domain = 100) — stated by the card caption either way.
  const radar = gridRelativeRadar(drivers.length > 0 ? drivers : [driver])[code];

  return (
    <main className="mx-auto max-w-7xl p-6">
      <ProfileHeader driver={driver} />
      {fp.flags.includes('STALE_DATA') ? (
        <p data-testid="stale" className="mb-3 font-mono text-[10px] uppercase tracking-[0.14em] text-accent-deep">
          Showing cached data.
        </p>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[1fr_1.15fr_0.95fr]">
        <StyleProfileCard radar={radar} colour={colour} code={code} />
        <SignalWeightsCard
          code={code}
          shapSigned={fp.data.shap_signed ?? {}}
          nlDescription={fp.data.nl_description ?? ''}
          signature={driver.signature ?? []}
        />
        <div className="space-y-4">
          <FactsCard driver={driver} stability={fp.data.stability} />
          <section className="rounded-md border border-border bg-card p-4" data-testid="setup-card">
            <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
              Setup recommendation
            </h2>
            {setupRes.ok ? (
              <SetupList setup={setupRes.data} />
            ) : setupRes.code === 'NOT_FOUND' ? (
              <p className="font-mono text-[11px] text-text-secondary">
                No setup data for {DEFAULT_CIRCUIT}.
              </p>
            ) : (
              // A real outage (5xx/DB/network) is an error, NOT honest absence (failsafe HIGH-3).
              <PageError code={setupRes.code} detail={setupRes.detail} />
            )}
          </section>
        </div>
      </div>

      <div className="mt-4">
        {styleRes.ok ? (
          <StyleMapCard style={styleRes.data} summaries={drivers} highlight={code} />
        ) : (
          <section className="rounded-md border border-border bg-card p-4">
            <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
              Style map
            </h2>
            <div className="mt-2">
              <PageError code={styleRes.code} detail={styleRes.detail} />
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
