import { EmptyState } from '@/components/ds/EmptyState';
import type { DriverSummary } from '@/lib/api/types';
import { datasetSummary, orderByIdentification } from '@/lib/design/overview';
import { gridRelativeRadar } from '@/lib/design/radar';

import { OverviewCard } from './OverviewCard';

// Screen 1 — Driver Overview. Pure render over the SSR'd /drivers payload: every number on screen
// (title counts, card stats, radar shapes) derives from that one response, so there is no client
// fetch fan-out and no loading choreography. Ordering and radar scaling are lib functions with
// their own tests.
export function OverviewGrid({
  initialDrivers,
  animate = true,
}: {
  initialDrivers: DriverSummary[];
  animate?: boolean;
}) {
  if (initialDrivers.length === 0) {
    return <EmptyState message="No driver fingerprints yet. Run an ingestion batch first." />;
  }

  const drivers = orderByIdentification(initialDrivers);
  const radar = gridRelativeRadar(drivers);
  const summary = datasetSummary(drivers);

  const subtitle = [
    `${summary.nDrivers} behavioural fingerprint${summary.nDrivers === 1 ? '' : 's'}`,
    summary.nLaps != null ? `${summary.nLaps.toLocaleString('en-GB')} clean laps of telemetry` : null,
    summary.seasonSpan ? `${summary.seasonSpan[0]} to ${summary.seasonSpan[1]} seasons` : null,
  ]
    .filter(Boolean)
    .join(', ');

  return (
    <div>
      <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-[34px] font-bold uppercase leading-none tracking-[0.03em] text-text-primary">
            Driver Overview
          </h1>
          <p className="mt-2 text-[13px] text-text-secondary">{subtitle}.</p>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
          Ordered by ID rate
        </p>
      </header>

      <div
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
        data-testid="overview-grid"
      >
        {drivers.map((d) => (
          <OverviewCard key={d.code} driver={d} radar={radar[d.code]} animate={animate} />
        ))}
      </div>

      {/* What the numbers mean — the card stays scannable, the definitions live once, down here. */}
      <p className="mt-6 max-w-5xl font-mono text-[10px] leading-relaxed text-text-secondary">
        Radar axes show where the classifier&apos;s attention concentrates per driver (mean |SHAP|
        per domain), scaled to the grid maximum on each axis. Key signals are the three features the
        model leans on most, with direction from the driver&apos;s rank against the grid.{' '}
        <span className="text-text-primary">Laps</span> counts clean laps in the training
        matrix. <span className="text-text-primary">Consistency</span> is the median within-session
        lap-time spread. <span className="text-text-primary">ID rate</span> is how reliably the
        classifier identifies the driver on circuits it never trained on (held-out F1).
      </p>
    </div>
  );
}
