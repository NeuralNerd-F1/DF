// Pure, jsdom-free logic for Screen 4 (Comparison). Kept out of the components so the derivations
// are unit-tested directly (the lib/replay/replay.ts precedent).
//
// WHY scaling lives here: /compare serves each radar as raw mean-|SHAP| magnitudes (unbounded —
// throttle can be 3.3), but FingerprintRadar/DeltaRadar plot a 0–1 radial and CLAMP. Feeding raw
// magnitudes rim-clamps most axes into an identical hexagon — the bug the overview redesign fixed
// with per-axis scaling. For a comparison the two radars must share ONE scale per axis (else the
// shapes aren't comparable), so every axis is divided by the PAIR's max for that axis.
import { RADAR_DOMAINS } from '@/lib/api/types';

// Per-domain max over both drivers (>= 0; missing/non-finite values count as 0).
export function pairAxisMaxima(
  r1: Record<string, number>,
  r2: Record<string, number>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const d of RADAR_DOMAINS) {
    const a = Number.isFinite(r1[d]) ? Math.abs(r1[d] as number) : 0;
    const b = Number.isFinite(r2[d]) ? Math.abs(r2[d] as number) : 0;
    out[d] = Math.max(a, b);
  }
  return out;
}

// values[d] / maxima[d] per domain — 0 when the axis max is 0 (no divide-by-zero, no fabricated
// magnitude). By construction the result is 0–1 for both drivers under the SAME per-axis scale.
export function scaleByMaxima(
  values: Record<string, number>,
  maxima: Record<string, number>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const d of RADAR_DOMAINS) {
    const m = maxima[d] ?? 0;
    const v = Number.isFinite(values[d]) ? (values[d] as number) : 0;
    out[d] = m > 0 ? v / m : 0;
  }
  return out;
}

// The per-FEATURE signed SHAP delta: driver1's shap_signed minus driver2's, over the union of
// keys, non-finite entries dropped. Positive = the model leans on this feature MORE for driver 1.
// Both inputs are served vectors (/drivers/{code}/fingerprint.shap_signed) — this is presentation
// arithmetic over served data, not a re-derived model output.
export function featureShapDelta(
  s1: Record<string, number>,
  s2: Record<string, number>,
): Record<string, number> {
  const keys = new Set([...Object.keys(s1), ...Object.keys(s2)]);
  const out: Record<string, number> = {};
  for (const k of keys) {
    const a = Number.isFinite(s1[k]) ? (s1[k] as number) : 0;
    const b = Number.isFinite(s2[k]) ? (s2[k] as number) : 0;
    if (Number.isFinite(s1[k]) || Number.isFinite(s2[k])) out[k] = a - b;
  }
  return out;
}
