'use client';

import type { ReplayLap } from '@/lib/api/types';
import { FEATURE_LABELS } from '@/lib/design/features';

// The degradation alert banner (master §10.7). The route now forwards the F10 DegradationAlert
// as of each lap (div #74 closed), so the banner shows the REAL payload: alert confidence, feature
// drivers, laps since the threshold crossing, historical pit window. It renders from the lap the
// alert fires on and stays up for the rest of the replay (the payload advances with the prefix —
// laps_since_threshold_crossed grows); scrubbing back before the firing lap hides it (alert=null —
// the monitor knew nothing yet). No alert in the whole replay → never renders (nominal monitoring).
const MAX_DRIVER_CHIPS = 4;

export function AlertBanner({ lap }: { lap: ReplayLap | null }) {
  const alert = lap?.alert ?? null;
  if (!alert) return null;
  const active = lap?.alert_state === 'ALERT';
  const attribution = alert.feature_attribution ?? {};
  const drivers = (alert.top_features ?? []).slice(0, MAX_DRIVER_CHIPS);
  const window = alert.optimal_window_remaining_laps;

  return (
    <div
      role="alert"
      data-testid="alert-banner"
      data-active={active || undefined}
      className={`animate-slide-down rounded-md border border-l-4 border-border bg-card p-4 ${
        active ? 'border-l-alert-ink' : 'border-l-text-secondary'
      }`}
    >
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span
          className={`font-mono text-[13px] uppercase tracking-[0.14em] ${
            active ? 'text-alert-ink' : 'text-text-secondary'
          }`}
        >
          Behavioural deviation alert
        </span>
        <span className="font-mono text-xs text-text-primary" data-testid="alert-lap">
          Lap {alert.alert_lap}
        </span>
        <span
          className="rounded border border-border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.14em] text-text-primary"
          data-testid="alert-confidence"
        >
          {alert.alert_confidence} confidence
        </span>
        <span className="font-mono text-xs text-text-secondary" data-testid="alert-laps-since">
          {alert.laps_since_threshold_crossed} lap
          {alert.laps_since_threshold_crossed === 1 ? '' : 's'} since threshold crossed
        </span>
        <span className="font-mono text-xs text-text-secondary" data-testid="alert-pit-window">
          {window != null
            ? `historical pit window: ${window} lap${window === 1 ? '' : 's'} (median-based)`
            : 'no historical pit-window estimate'}
        </span>
        {!active ? (
          <span className="font-mono text-[10px] text-text-secondary" data-testid="alert-recovered">
            deviation back under τ, after an alert fired earlier in this stint
          </span>
        ) : null}
      </div>
      {drivers.length > 0 ? (
        <div className="mt-2 flex flex-wrap items-baseline gap-1.5" data-testid="alert-feature-drivers">
          <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
            Feature drivers
          </span>
          {drivers.map((f) => {
            const w = attribution[f];
            return (
              <span
                key={f}
                className="rounded border border-border px-1.5 py-0.5 font-mono text-[9px] text-text-primary"
                data-testid={`alert-driver-${f}`}
                title={`${f}, share of the deviation score attributed to this feature`}
              >
                {FEATURE_LABELS[f] ?? f}
                {typeof w === 'number' && Number.isFinite(w) ? (
                  <span className="ml-1 text-text-secondary">{w.toFixed(2)}</span>
                ) : null}
              </span>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
