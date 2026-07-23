// A clearly-SYNTHETIC 28-feature example vector for the manual prediction panel. Divergence #54:
// no endpoint yields a real per-session feature vector, so Screen 3's prediction is a manual tool —
// this is a labelled starting point the user edits, not real telemetry. The prediction itself is
// real (the backend classifier computes it). Keys F01..F28 mirror the master feature ids.
export const SAMPLE_FEATURE_VECTOR: Record<string, number> = {
  F01: 0.62, F02: 0.48, F03: 0.71, F04: 0.33, F05: 0.55, F06: 0.4, F07: 0.68,
  F08: 0.29, F09: 0.51, F10: 0.6, F11: 0.44, F12: 0.73, F13: 0.38, F14: 0.5,
  F15: 0.66, F16: 0.42, F17: 0.35, F18: 0.58, F19: 0.47, F20: 0.63, F21: 0.31,
  F22: 0.54, F23: 0.49, F24: 0.57, F25: 0.41, F26: 0.6, F27: 0.36, F28: 0.52,
};

export const SAMPLE_VECTOR_JSON = JSON.stringify(SAMPLE_FEATURE_VECTOR, null, 2);

// A synthetic 6-lap set for the Unknown Driver tab. 6 laps clears the 5-lap projection floor and
// (being < 30) demonstrates the honest SPARSE_PROJECTION state. Deterministic (fixed per-lap offset,
// no Math.random) so the default input and tests are reproducible.
export const SAMPLE_LAPS: Array<Record<string, number>> = Array.from({ length: 6 }, (_, lap) =>
  Object.fromEntries(
    Object.entries(SAMPLE_FEATURE_VECTOR).map(([k, v]) => [
      k,
      Math.round((v + (lap - 3) * 0.01) * 1000) / 1000,
    ]),
  ),
);

export const SAMPLE_LAPS_JSON = JSON.stringify(SAMPLE_LAPS, null, 2);
