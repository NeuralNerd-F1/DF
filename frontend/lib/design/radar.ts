import { RADAR_DOMAINS, type DriverSummary, type RadarDomain } from '@/lib/api/types';

// Three-letter axis tags for the mini radar — full domain names don't fit a 110px chart.
export const RADAR_MICRO_LABELS: Readonly<Record<RadarDomain, string>> = {
  throttle: 'THR',
  braking: 'BRK',
  speed: 'SPD',
  gear: 'GER',
  engine: 'ENG',
  cornering: 'COR',
};

// Grid-relative display scaling for the overview radars.
//
// radar_values are Σ mean |SHAP| per domain — unbounded magnitudes (throttle reaches ~3.3 on this
// grid). Rendering them into a chart whose rim is 1.0 silently clamps every strong domain flat
// against the rim, which reads as "identical maximal drivers" — a display lie. Dividing each
// domain by the grid's maximum for THAT domain puts every axis on "share of the grid's strongest",
// so shapes differ where the data differs and nothing clamps. The caption owns the claim: axes are
// scaled to the grid maximum, so radars are comparable ACROSS drivers per-axis, and the original
// magnitudes stay available untouched on the payload.
export function gridRelativeRadar(
  drivers: readonly Pick<DriverSummary, 'code' | 'radar_values'>[],
): Record<string, Record<RadarDomain, number>> {
  const max: Partial<Record<RadarDomain, number>> = {};
  for (const domain of RADAR_DOMAINS) {
    let m = 0;
    for (const d of drivers) {
      const v = d.radar_values?.[domain];
      if (typeof v === 'number' && Number.isFinite(v) && v > m) m = v;
    }
    max[domain] = m;
  }

  const out: Record<string, Record<RadarDomain, number>> = {};
  for (const d of drivers) {
    const scaled = {} as Record<RadarDomain, number>;
    for (const domain of RADAR_DOMAINS) {
      const v = d.radar_values?.[domain];
      const m = max[domain] ?? 0;
      // A domain nobody scores on (or a missing value) renders at the centre — an honest zero,
      // not a mid-scale guess.
      scaled[domain] =
        typeof v === 'number' && Number.isFinite(v) && m > 0 ? Math.max(0, v) / m : 0;
    }
    out[d.code] = scaled;
  }
  return out;
}
