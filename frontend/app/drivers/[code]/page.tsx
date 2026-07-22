import Link from 'next/link';

import { CornerTabs } from '@/components/driver-profile/CornerTabs';
import { MultiSeasonPanel } from '@/components/driver-profile/MultiSeasonPanel';
import { NlDescription } from '@/components/driver-profile/NlDescription';
import { ProfileRadar } from '@/components/driver-profile/ProfileRadar';
import { SetupPanel } from '@/components/driver-profile/SetupPanel';
import { ShapBeeswarmPanel } from '@/components/driver-profile/ShapBeeswarmPanel';
import { PageError } from '@/components/ds/PageError';
import { getDrivers, getFingerprint, getSetup } from '@/lib/api/endpoints';
import { RADAR_DOMAINS, type DriverSummary } from '@/lib/api/types';
import { DEFAULT_CIRCUIT } from '@/lib/design/circuits';

export const dynamic = 'force-dynamic';

// Field mean per domain over the 8 drivers (F16 spec div #51) — client-derived display value. A
// domain with NO numeric data is OMITTED (not defaulted to 0) so the tooltip hides its mean line
// rather than showing a fabricated "field mean 0.00" (failsafe MEDIUM).
function fieldMeanOf(drivers: DriverSummary[]): Record<string, number> {
  const mean: Record<string, number> = {};
  for (const k of RADAR_DOMAINS) {
    const xs = drivers
      .map((d) => d.radar_values?.[k])
      .filter((v): v is number => typeof v === 'number');
    if (xs.length) mean[k] = xs.reduce((a, b) => a + b, 0) / xs.length;
  }
  return mean;
}

export default async function DriverProfilePage({ params }: { params: { code: string } }) {
  const code = params.code;
  const [fp, driversRes, setupRes] = await Promise.all([
    getFingerprint(code),
    getDrivers(),
    getSetup(code, DEFAULT_CIRCUIT),
  ]);

  if (!fp.ok) {
    return (
      <main className="mx-auto max-w-6xl p-6">
        <Link href="/" className="font-mono text-xs text-accent hover:underline">
          ← Command Centre
        </Link>
        <div className="mt-4">
          <PageError code={fp.code} detail={fp.detail} />
        </div>
      </main>
    );
  }

  // Only compute a field mean when there is a real, non-empty field to average (failsafe MEDIUM):
  // otherwise the tooltip shows no mean line rather than a fabricated one.
  const fieldMean =
    driversRes.ok && driversRes.data.length > 0 ? fieldMeanOf(driversRes.data) : undefined;

  return (
    <main className="mx-auto max-w-6xl p-6">
      <Link href="/" className="font-mono text-xs text-accent hover:underline">
        ← Command Centre
      </Link>
      <h1 className="mb-4 mt-2 font-mono text-2xl text-text-primary">{code}</h1>
      {fp.flags.includes('STALE_DATA') ? (
        <p data-testid="stale" className="mb-2 text-[10px] text-accent-deep">
          Showing cached data.
        </p>
      ) : null}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Left — the data-backed panel */}
        <div className="space-y-4">
          <ProfileRadar values={fp.data.shap_radar ?? {}} fieldMean={fieldMean} />
          <NlDescription text={fp.data.nl_description ?? ''} />
          {setupRes.ok ? (
            <SetupPanel setup={setupRes.data} />
          ) : setupRes.code === 'NOT_FOUND' ? (
            <p className="text-xs text-text-secondary">No setup data for {DEFAULT_CIRCUIT}.</p>
          ) : (
            // A real outage (5xx/DB/network) is an error, NOT honest absence (failsafe HIGH-3).
            <PageError code={setupRes.code} detail={setupRes.detail} />
          )}
        </div>
        {/* Right — multi-season + honest-unavailable per-lap / corner views */}
        <div className="space-y-4">
          <MultiSeasonPanel stability={fp.data.stability} />
          <ShapBeeswarmPanel laps={[]} />
          <CornerTabs />
        </div>
      </div>
    </main>
  );
}
