// Plain-English vocabulary for the 28 feature IDs (master §4.2) — the layer that turns
// "F02 · 1.14" into a sentence a race engineer would say. Two rules keep the copy honest:
//
// 1. The FEATURE the card names comes from the champion's per-driver mean |SHAP| — "the model
//    watches this channel for this driver". That is a magnitude and carries NO direction.
// 2. The DIRECTION word comes from the driver's grid percentile on that feature's raw mean —
//    measured separately in the profiles artefact. High rank → the high phrase, low rank → the
//    low phrase, middle (or no rank) → the neutral "distinctive …" phrase, which claims nothing.
//
// So every word on the card traces to one of two measured numbers, and the two are never blended.
import type { DriverSignal } from '@/lib/api/types';

// Short noun labels (tooltips, axis captions, the neutral phrase).
export const FEATURE_LABELS: Readonly<Record<string, string>> = {
  F01: 'throttle application',
  F02: 'throttle modulation',
  F03: 'throttle ramp rate',
  F04: 'full-throttle share',
  F05: 'throttle-lift share',
  F06: 'peak brake pressure',
  F07: 'braking duration',
  F08: 'brake release',
  F09: 'corner-entry speed',
  F10: 'brake-throttle overlap',
  F11: 'mid-corner speed',
  F12: 'speed spread',
  F13: 'slow-phase speed',
  F14: 'top-end speed',
  F15: 'gear-change rate',
  F16: 'gear-hold time',
  F17: 'high-RPM share',
  F18: 'RPM spread',
  F19: 'lateral load',
  F20: 'peak cornering load',
  F21: 'steering corrections',
  F22: 'throttle-speed coupling',
  F23: 'brake-speed coupling',
  F24: 'brake-to-throttle gap',
  F25: 'corner-entry balance',
  F26: 'lap-phase throttle shift',
  F27: 'tyre-age throttle drift',
  F28: 'tyre-age brake drift',
};

// Directional phrasing per feature. `high` reads for a grid rank in the top band, `low` for the
// bottom band. Written per-feature because "high brake release" is not English — each pair names
// what a high/low value of THAT channel physically looks like.
const SIGNAL_PHRASES: Readonly<Record<string, { high: string; low: string }>> = {
  F01: { high: 'Heavy throttle application', low: 'Sparing throttle application' },
  F02: { high: 'Strongly modulated throttle', low: 'Steady throttle trace' },
  F03: { high: 'Aggressive throttle ramps', low: 'Progressive throttle ramps' },
  F04: { high: 'Long full-throttle stretches', low: 'Short full-throttle stretches' },
  F05: { high: 'Frequent throttle lifts', low: 'Rare throttle lifts' },
  F06: { high: 'Hard peak braking', low: 'Gentle peak braking' },
  F07: { high: 'Long braking zones', low: 'Short braking zones' },
  F08: { high: 'Snap brake release', low: 'Gradual brake release' },
  F09: { high: 'High corner-entry speed', low: 'Cautious corner entry' },
  F10: { high: 'Overlapped brake and throttle', low: 'Clean brake-throttle separation' },
  F11: { high: 'High mid-corner speed', low: 'Low mid-corner speed' },
  F12: { high: 'Wide speed range', low: 'Narrow speed range' },
  F13: { high: 'High slow-phase speed', low: 'Deep slow-corner speeds' },
  F14: { high: 'High top-end speed', low: 'Modest top-end speed' },
  F15: { high: 'Busy gear changes', low: 'Economical gear changes' },
  F16: { high: 'Long gear holds', low: 'Short gear holds' },
  F17: { high: 'Sustained high-RPM running', low: 'Conservative RPM usage' },
  F18: { high: 'Wide RPM swings', low: 'Steady RPM band' },
  F19: { high: 'High sustained lateral load', low: 'Low sustained lateral load' },
  F20: { high: 'High peak cornering load', low: 'Soft peak cornering load' },
  F21: { high: 'Frequent steering corrections', low: 'Minimal steering corrections' },
  F22: { high: 'Tight throttle-speed coupling', low: 'Loose throttle-speed coupling' },
  F23: { high: 'Tight brake-speed coupling', low: 'Loose brake-speed coupling' },
  F24: { high: 'Unhurried brake-to-throttle switch', low: 'Instant brake-to-throttle switch' },
  F25: { high: 'Entry-biased corner speed', low: 'Exit-biased corner speed' },
  F26: { high: 'Shifting throttle over the lap', low: 'Constant throttle over the lap' },
  F27: { high: 'Strong tyre-age throttle drift', low: 'Minimal tyre-age throttle drift' },
  F28: { high: 'Strong tyre-age brake drift', low: 'Minimal tyre-age brake drift' },
};

// Percentile bands for the direction words. Outside the top/bottom third the neutral phrase is
// used — a rank of 0.5 does not justify "high" or "low" anything.
export const SIGNAL_HIGH_BAND = 2 / 3;
export const SIGNAL_LOW_BAND = 1 / 3;

// The card sentence for one signal. Unknown feature IDs fall back to the raw ID — a future F29
// renders as "Distinctive F29", never a crash or a silent drop.
export function signalText(signal: Pick<DriverSignal, 'feature' | 'percentile'>): string {
  const label = FEATURE_LABELS[signal.feature] ?? signal.feature;
  const phrases = SIGNAL_PHRASES[signal.feature];
  const p = signal.percentile;
  if (phrases && p !== null && p !== undefined) {
    if (p >= SIGNAL_HIGH_BAND) return phrases.high;
    if (p <= SIGNAL_LOW_BAND) return phrases.low;
  }
  return `Distinctive ${label}`;
}

// Hover provenance for a signal — the measured numbers behind the sentence.
export function signalDetail(signal: DriverSignal): string {
  const label = FEATURE_LABELS[signal.feature] ?? signal.feature;
  const rank =
    signal.percentile === null || signal.percentile === undefined
      ? 'no grid rank'
      : `grid rank ${Math.round(signal.percentile * 100)}%`;
  return `${signal.feature} ${label}, model attribution ${signal.attribution.toFixed(2)} · ${rank}`;
}
