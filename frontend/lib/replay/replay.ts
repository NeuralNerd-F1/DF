// Pure, jsdom-free logic for Screen 6 (Historical Race Replay Monitor). Kept out of the components
// so the derivations are unit-tested directly (the F18 lib/style-space/styleSpace.ts precedent).
import type { ConfidenceState, ReplayFingerprintSnapshot, ReplayLap } from '@/lib/api/types';

// Alert-state → visual tone. The gauge/timeline colour off the monitor's alert_state (div #72),
// NOT a numeric τ compare — τ (threshold) is dropped by the route. The semantic (alert) is kept
// separate in code AND in colour: monitor tones map to the operational token family.
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

// Colour classes per tone (the colours themselves live once in tailwind.config.js — no literal
// here). Monitor states are OPERATIONAL tokens, never the confidence (assert/epistemic) family:
// nominal sits on the baseline grey and escalation is by intensity, not hue.
export const ALERT_FILL_CLASS: Record<AlertTone, string> = {
  nominal: 'fill-threshold',
  watch: 'fill-watch',
  alert: 'fill-alert-ink',
  idle: 'fill-unavailable',
};
export const ALERT_STROKE_CLASS: Record<AlertTone, string> = {
  nominal: 'stroke-threshold',
  watch: 'stroke-watch',
  alert: 'stroke-alert-ink',
  idle: 'stroke-unavailable',
};
export const ALERT_TEXT_CLASS: Record<AlertTone, string> = {
  nominal: 'text-threshold',
  watch: 'text-watch',
  alert: 'text-alert-ink',
  idle: 'text-text-tertiary',
};

// The stint's τ — LapDeviation.threshold is constant per stint but null on BASELINE/SKIPPED laps,
// so take the first finite value anywhere in the trace. null when the monitor never activated.
export function stintThreshold(laps: readonly ReplayLap[]): number | null {
  for (const l of laps) {
    const t = l.threshold;
    if (typeof t === 'number' && Number.isFinite(t)) return t;
  }
  return null;
}

// The max finite deviation_score across the replay; falls back to 1 so an all-null replay still
// renders a valid (empty) gauge instead of dividing by zero.
export function deviationMax(laps: readonly ReplayLap[]): number {
  let m = 0;
  for (const l of laps) {
    const s = l.deviation_score;
    if (typeof s === 'number' && Number.isFinite(s) && s > m) m = s;
  }
  return m > 0 ? m : 1;
}

// Gauge/timeline scale max. τ now ships per lap (div #72/#73 closed), so the scale is anchored to
// the REAL threshold: headroom above τ keeps the dashed line inside the chart even before any lap
// crosses it, and the observed stint range still wins when deviations overshoot τ.
export const THRESHOLD_HEADROOM = 1.25;

export function gaugeMax(laps: readonly ReplayLap[]): number {
  const tau = stintThreshold(laps);
  const observed = deviationMax(laps);
  return tau != null ? Math.max(observed, tau * THRESHOLD_HEADROOM) : observed;
}

// The fingerprint drift trail — every reached lap's REAL F08 2D projection (master §10.7 "the
// fingerprint evolving in real time"). Laps without a snapshot (or non-finite coords) are simply
// absent from the trail, never fabricated.
export type DriftPoint = { x: number; y: number; index: number };

export function driftTrail(laps: readonly ReplayLap[], uptoIndex: number): DriftPoint[] {
  const limit = Math.min(uptoIndex, laps.length - 1);
  const out: DriftPoint[] = [];
  for (let i = 0; i <= limit; i++) {
    const snap = narrowSnapshot(laps[i]?.fingerprint_snapshot);
    const c = snap?.coords_2d;
    const x = c?.[0];
    const y = c?.[1];
    if (typeof x === 'number' && Number.isFinite(x) && typeof y === 'number' && Number.isFinite(y)) {
      out.push({ x, y, index: i });
    }
  }
  return out;
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
