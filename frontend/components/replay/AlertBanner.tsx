'use client';

import type { ReplayLap } from '@/lib/api/types';
import { firstAlertLap, lapsInAlert } from '@/lib/replay/replay';

// The degradation alert banner (master §10.7, div #74). Fires when a REACHED lap is ALERT: a
// full-width banner (animates down) showing the alert lap number + client-derived laps-in-alert
// (counting the ALERT run in the state sequence). The route drops res.alert, so the alert's
// confidence / feature-drivers / recommended-pit-window are DEFERRED — shown as an explicit note,
// never fabricated. No ALERT reached → no banner (honest; monitoring nominal). Named seam: forward
// DegradationAlert per alert lap.
type Props = { laps: readonly ReplayLap[]; reachedIndex: number };

export function AlertBanner({ laps, reachedIndex }: Props) {
  const alertLap = firstAlertLap(laps, reachedIndex);
  if (!alertLap) return null;
  const inAlert = lapsInAlert(laps, reachedIndex);

  return (
    <div
      role="alert"
      data-testid="alert-banner"
      className="animate-slide-down rounded-lg border border-l-4 border-border border-l-accent bg-card p-4"
    >
      <div className="flex flex-wrap items-baseline gap-3">
        <span className="font-mono text-sm uppercase text-accent">Degradation Alert</span>
        <span className="font-mono text-xs text-text-primary" data-testid="alert-lap">
          Lap {alertLap.lap_number}
        </span>
        <span className="font-mono text-xs text-text-secondary" data-testid="alert-laps-in-alert">
          {inAlert} lap{inAlert === 1 ? '' : 's'} in alert
        </span>
      </div>
      <p className="mt-2 font-mono text-[10px] text-text-secondary" data-testid="alert-detail-note">
        Alert detail (confidence, feature drivers, pit window) needs the replay alert-payload seam.
      </p>
    </div>
  );
}
