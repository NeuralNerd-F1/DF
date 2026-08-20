import Link from 'next/link';

import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import type { DriverSummary, RadarDomain } from '@/lib/api/types';
import { DURATION, STAGGER } from '@/lib/design/animation';
import { driverNumber } from '@/lib/design/drivers';
import { signalDetail, signalText } from '@/lib/design/features';
import { teamColour } from '@/lib/design/teams';

// Surname in caps, the reference treatment. full_name comes from the DB; a driver row without one
// falls back to the code rather than inventing a name.
function displayName(driver: DriverSummary): string {
  const last = driver.full_name?.trim().split(/\s+/).slice(-1)[0];
  return (last ?? driver.code).toUpperCase();
}

function StatValue({
  children,
  missing,
}: {
  children: React.ReactNode;
  missing?: boolean;
}) {
  return (
    <dd
      className={`font-mono text-[13px] tabular-nums ${missing ? 'text-text-secondary' : 'text-text-primary'}`}
    >
      {children}
    </dd>
  );
}

// One driver on the overview grid. Everything on the card is measured: the radar is the champion's
// per-domain SHAP attribution (grid-scaled by the parent), the signals are its top-3 attributed
// features with grid-rank direction words, and the stats row is the training matrix + held-out
// evaluation. A missing measurement renders an explicit dash, never a filler number.
export function OverviewCard({
  driver,
  radar,
  animate = true,
}: {
  driver: DriverSummary;
  radar: Record<RadarDomain, number> | undefined;
  animate?: boolean;
}) {
  const colour = teamColour(driver.team);
  const number = driverNumber(driver.code);
  const profile = driver.profile ?? null;
  const signals = driver.signature ?? [];
  const ruleId = `rule-${driver.code}`;

  return (
    <Link
      href={`/drivers/${driver.code}`}
      className="group block focus:outline-none focus:ring-1 focus:ring-accent"
      data-testid={`overview-card-${driver.code}`}
    >
      <div className="overflow-hidden rounded-md border border-border bg-card transition-colors group-hover:border-text-secondary">
        {/* Team rule — solid at the left, fading right (SVG gradient; per-card id). */}
        <svg viewBox="0 0 100 3" preserveAspectRatio="none" className="h-[3px] w-full" aria-hidden="true">
          <defs>
            {/* stop values avoid 0.65/0.85 — the design-guard test reserves those literals for
                catching a re-derived confidence gate */}
            <linearGradient id={ruleId} x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stopColor={colour} />
              <stop offset="0.6" stopColor={colour} stopOpacity={0.9} />
              <stop offset="1" stopColor={colour} stopOpacity={0.15} />
            </linearGradient>
          </defs>
          <rect width={100} height={3} fill={`url(#${ruleId})`} />
        </svg>

        <div className="p-4">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              {/* Race number — team-coloured display type via SVG fill (no inline style). */}
              <svg
                viewBox={`0 0 ${number.length * 26 + 4} 40`}
                width={number.length * 26 + 4}
                height={40}
                aria-hidden="true"
                className="-ml-0.5 block"
              >
                <text
                  x={0}
                  y={34}
                  fill={colour}
                  className="font-display text-[42px] font-bold"
                  data-testid="driver-number"
                >
                  {number}
                </text>
              </svg>
              <h2 className="mt-1 truncate font-display text-[21px] font-semibold leading-none tracking-[0.02em] text-text-primary">
                {displayName(driver)}
              </h2>
              <p className="mt-1 truncate font-mono text-[10px] uppercase tracking-[0.14em] text-text-secondary">
                {driver.team ?? 'Team unknown'}
              </p>
            </div>
            <FingerprintRadar
              values={radar ?? {}}
              size={116}
              showLabels={false}
              microLabels
              colour={colour}
              animate={animate}
              staggerMs={STAGGER.screen1Axis}
              durationMs={DURATION.screen1Bloom}
              ariaLabel={`${driver.code} fingerprint, axes scaled to grid maximum`}
              className="-mr-1 -mt-1 shrink-0"
            />
          </div>

          {/* Key signals — top-3 model-attributed features, direction from grid rank. */}
          <ul className="mt-3 space-y-1.5 border-t border-border pt-3">
            {signals.length > 0 ? (
              signals.map((s) => (
                <li
                  key={s.feature}
                  className="flex items-center gap-2 font-mono text-[11px] leading-tight text-text-primary"
                  title={signalDetail(s)}
                >
                  <svg width={5} height={5} aria-hidden="true" className="shrink-0">
                    <rect width={5} height={5} fill={colour} />
                  </svg>
                  {signalText(s)}
                </li>
              ))
            ) : (
              <li
                className="font-mono text-[11px] text-text-secondary"
                title="This fingerprint predates the SHAP attribution fill, so no model attribution is stored"
              >
                No attribution available
              </li>
            )}
          </ul>

          {/* Dataset + evaluation facts. Dash = not measured, never zero. */}
          <dl className="mt-3 grid grid-cols-3 gap-2 border-t border-border pt-3">
            <div title="Clean laps of this driver in the training matrix">
              <dt className="font-mono text-[8.5px] uppercase tracking-[0.18em] text-text-secondary">
                Laps
              </dt>
              <StatValue missing={!profile}>
                {profile ? profile.laps_analysed.toLocaleString('en-GB') : '—'}
              </StatValue>
            </div>
            <div title="Median within-session lap-time spread over clean laps. Lower means more repeatable.">
              <dt className="font-mono text-[8.5px] uppercase tracking-[0.18em] text-text-secondary">
                Consistency
              </dt>
              <StatValue missing={profile?.lap_time_consistency_s == null}>
                {profile?.lap_time_consistency_s != null
                  ? `±${profile.lap_time_consistency_s.toFixed(2)}s`
                  : '—'}
              </StatValue>
            </div>
            <div title="How reliably the classifier identifies this driver on circuits it never trained on (held-out F1)">
              <dt className="font-mono text-[8.5px] uppercase tracking-[0.18em] text-text-secondary">
                ID rate
              </dt>
              <StatValue missing={profile?.identification_f1 == null}>
                {profile?.identification_f1 != null
                  ? `${Math.round(profile.identification_f1 * 100)}%`
                  : '—'}
              </StatValue>
            </div>
          </dl>
        </div>
      </div>
    </Link>
  );
}
