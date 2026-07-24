// Pure, jsdom-free logic for Screen 6 (Historical Race Replay Monitor). Kept out of the components
// so the derivations are unit-tested directly (the F18 lib/style-space/styleSpace.ts precedent).
import type { ConfidenceState, ReplayFingerprintSnapshot, ReplayLap } from '@/lib/api/types';

// Alert-state → visual tone. The gauge/timeline colour off the monitor's alert_state (div #72),
// NOT a numeric τ compare — τ (threshold) is dropped by the route. The semantic (alert) is kept
// separate in code even though the colours reuse the existing green/amber/red confidence tokens.
export type AlertTone = 'nominal' | 'watch' | 'alert' | 'idle';

export function alertTone(state: string | null | undefined): AlertTone {
  switch (state) {
    case 'NOMINAL':
      return 'nominal';
    case 'WATCH':
      return 'watch';
    case 'ALERT':
      return 'alert';
    default:
      // INACTIVE / BASELINE / SKIPPED / PENDING / HALTED → idle grey (no score to judge)
      return 'idle';
  }
}

// Colour classes per tone (the colours themselves live once in tailwind.config.js — no literal here).
export const ALERT_FILL_CLASS: Record<AlertTone, string> = {
  nominal: 'fill-confidence-high',
  watch: 'fill-confidence-medium',
  alert: 'fill-confidence-low',
  idle: 'fill-border',
};
export const ALERT_STROKE_CLASS: Record<AlertTone, string> = {
  nominal: 'stroke-confidence-high',
  watch: 'stroke-confidence-medium',
  alert: 'stroke-confidence-low',
  idle: 'stroke-border',
};
export const ALERT_TEXT_CLASS: Record<AlertTone, string> = {
  nominal: 'text-confidence-high',
  watch: 'text-confidence-medium',
  alert: 'text-confidence-low',
  idle: 'text-text-secondary',
};

export function hasAnyAlert(laps: readonly ReplayLap[]): boolean {
  return laps.some((l) => l.alert_state === 'ALERT');
}

// The client-derived "laps in alert" (div #74): the CONTIGUOUS ALERT run ending at reachedIndex. If
// the reached lap is not ALERT, 0. Walks backward through the state sequence — no server field used.
export function lapsInAlert(laps: readonly ReplayLap[], reachedIndex: number): number {
  if (reachedIndex < 0 || reachedIndex >= laps.length) return 0;
  if (laps[reachedIndex]?.alert_state !== 'ALERT') return 0;
  let n = 0;
  for (let i = reachedIndex; i >= 0 && laps[i]?.alert_state === 'ALERT'; i--) n++;
  return n;
}

// The FIRST reached lap whose state is ALERT — the banner headline. null if none reached yet.
export function firstAlertLap(laps: readonly ReplayLap[], reachedIndex: number): ReplayLap | null {
  const limit = Math.min(reachedIndex, laps.length - 1);
  for (let i = 0; i <= limit; i++) {
    if (laps[i]?.alert_state === 'ALERT') return laps[i] ?? null;
  }
  return null;
}

// Gauge/timeline max = the client-derived stint deviation range (div #72), since τ is not exposed.
// Uses the max finite deviation_score across the replay; falls back to 1 so an all-null replay still
// renders a valid (empty) gauge instead of dividing by zero.
export function deviationMax(laps: readonly ReplayLap[]): number {
  let m = 0;
  for (const l of laps) {
    const s = l.deviation_score;
    if (typeof s === 'number' && Number.isFinite(s) && s > m) m = s;
  }
  return m > 0 ? m : 1;
}

// Narrow the open fingerprint_snapshot dict to the three keys it carries (div #75). confidence_state
// is the ONE real per-lap datum; coords are F08 5D/2D, NOT a 6-domain radar — never fabricate one.
export function narrowSnapshot(snap: unknown): ReplayFingerprintSnapshot | null {
  if (!snap || typeof snap !== 'object') return null;
  const s = snap as Record<string, unknown>;
  const cs = s.confidence_state;
  return {
    coords_5d: Array.isArray(s.coords_5d) ? (s.coords_5d as number[]) : null,
    coords_2d: Array.isArray(s.coords_2d) ? (s.coords_2d as number[]) : null,
    confidence_state: typeof cs === 'string' ? (cs as ConfidenceState) : null,
  };
}
