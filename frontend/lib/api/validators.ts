// Runtime shape-guards for the responses the frontend consumes. Validated at the fetch boundary
// so a malformed body becomes UI_SCHEMA_MISMATCH instead of a silent bad render (F16 spec §4.0-a).
// These guard the load-bearing fields and `.passthrough()` the rest — the compile-time contract is
// the GENERATED types (lib/api/types.ts); a backend rename is caught by the pytest drift guard.
import { z } from 'zod';

const radar = z.record(z.number());

// One feature the champion attributes to a driver. `percentile` is nullish because the profiles
// artefact may carry no grid rank for that feature — the card then names it without a direction.
const zDriverSignal = z
  .object({
    feature: z.string(),
    attribution: z.number(),
    percentile: z.number().nullish(),
  })
  .passthrough();

// Dataset + evaluation facts. Every optional number is nullish, NOT defaulted to 0: "not measured"
// and "measured as zero" must stay distinguishable all the way to the pixel.
const zDriverProfile = z
  .object({
    laps_analysed: z.number(),
    seasons: z.array(z.number()).optional(),
    n_circuits: z.number().optional(),
    lap_time_consistency_s: z.number().nullish(),
    identification_f1: z.number().nullish(),
  })
  .passthrough();

export const zDriverSummary = z
  .object({
    id: z.string(),
    code: z.string(),
    team: z.string().nullish(),
    full_name: z.string().nullish(),
    radar_values: radar.optional(),
    signature: z.array(zDriverSignal).optional(),
    profile: zDriverProfile.nullish(),
  })
  .passthrough();
export const zDriverSummaryList = z.array(zDriverSummary);

export const zStability = z
  .object({
    driver_id: z.string(),
    stability_score: z.number().nullish(),
    pairwise_js: z.record(z.number()).optional(),
    seasons_available: z.number(),
    partial: z.boolean(),
  })
  .passthrough();

export const zFingerprint = z
  .object({
    driver_id: z.string(),
    vector_5d: z.array(z.number()),
    shap_radar: radar.optional(),
    // Signed per-feature attribution — may be absent on a checkout without the profiles artefact;
    // the screen must then omit the confirms/argues panel, so absence stays representable.
    shap_signed: z.record(z.number()).optional(),
    nl_description: z.string().optional(),
    stability: zStability.nullish(),
  })
  .passthrough();

export const zSetupParam = z
  .object({
    name: z.string(),
    kind: z.string(),
    value: z.union([z.number(), z.string()]).nullish(),
    value_range: z.array(z.number()).nullish(),
    category: z.string().nullish(),
    confidence: z.number(),
    rating: z.string(),
    unit: z.string().nullish(),
    fia_min: z.number().nullish(),
    fia_max: z.number().nullish(),
    constraint_flags: z.array(z.string()).optional(),
    shap_top_features: z.array(z.string()).optional(),
    shap_available: z.boolean().optional(),
  })
  .passthrough();
export const zSetup = z
  .object({
    driver: z.string(),
    circuit: z.string(),
    parameters: z.array(zSetupParam),
    consistency_warning: z.boolean().optional(),
    monaco_override: z.boolean().optional(),
    approximation_disclaimer: z.string(),
    model_version: z.string(),
  })
  .passthrough();

export const zStyleDriver = z
  .object({
    id: z.string(),
    code: z.string(),
    coords_2d: z.array(z.number()),
    coords_5d: z.array(z.number()),
    season: z.string().nullish(),
  })
  .passthrough();
export const zStyleSpace = z
  .object({
    // both optional to match the generated types — a valid empty style space omits them (avoids a
    // false UI_SCHEMA_MISMATCH; code-review #7). Consumed by F18, not F16a.
    drivers: z.array(zStyleDriver).optional(),
    unknown_drivers: z.array(zStyleDriver).optional(),
  })
  .passthrough();

export const zCompare = z
  .object({
    driver_1_radar: radar,
    driver_2_radar: radar,
    delta_shap: radar.optional(),
    most_distinctive_features: z.array(z.string()).optional(),
    compatibility_delta: z.number(),
  })
  .passthrough();

export const zHealth = z
  .object({
    status: z.string(),
    models_loaded: z.record(z.boolean()).optional(),
    db_connected: z.boolean(),
    cache_connected: z.boolean().nullish(),
    version: z.string(),
  })
  .passthrough();

// --- F17 Screen 3 (Ingestion & Prediction) --------------------------------------------------------

// POST /predict (F06b + F07). shap_values is a signed feature-id -> contribution map; top_3 optional.
export const zTopK = z.object({ driver: z.string(), probability: z.number() }).passthrough();
export const zPredict = z
  .object({
    predicted_driver: z.string().nullish(),
    confidence: z.number(),
    confidence_state: z.string().nullish(),
    top_3: z.array(zTopK).optional(),
    // Full calibrated distribution — optional so older payloads (and caches) stay valid.
    class_probabilities: z.record(z.number()).optional(),
    shap_values: z.record(z.number()).nullish(),
    shap_available: z.boolean().optional(),
    model_version: z.string(),
  })
  .passthrough();

// GET /segments — curated held-out sample laps for the Prediction screen picker. Features are
// nullable per-lap raw values; true_driver is the matrix ground truth used to grade the answer.
export const zSegment = z
  .object({
    id: z.string(),
    label: z.string(),
    circuit: z.string(),
    season: z.number(),
    session: z.string().optional(),
    lap_number: z.number().optional(),
    true_driver: z.string(),
    features: z.record(z.number().nullable()).optional(),
  })
  .passthrough();
export const zSegmentList = z.array(zSegment);

// GET /tyres — per-driver/compound stint degradation. Curve points carry their own lap counts;
// cliff_age is nullish because most corrected curves are near-linear and honestly cliff-free.
const zTyreCurvePoint = z
  .object({ age: z.number(), delta_s: z.number(), q25: z.number(), q75: z.number(), n: z.number() })
  .passthrough();
const zTyreCompound = z
  .object({
    n_laps: z.number(),
    n_stints: z.number(),
    median_stint_len: z.number(),
    max_stint_len: z.number(),
    deg_slope_s_per_lap: z.number(),
    total_delta_s: z.number(),
    fuel_slope_s_per_lap: z.number().nullish(),
    cliff_age: z.number().nullish(),
    curve: z.array(zTyreCurvePoint).optional(),
    f27_median: z.number().nullish(),
    f28_median: z.number().nullish(),
  })
  .passthrough();
export const zTyreWear = z
  .object({ drivers: z.record(z.record(zTyreCompound)).optional() })
  .passthrough();

// POST /fingerprint/unknown (F08). radar_values/style_description are returned EMPTY by the route
// (div #59) — guarded but not relied upon. nearest_neighbours/risk_flags/warnings drive the UI.
export const zNeighbour = z
  .object({
    driver: z.string(),
    cosine_similarity: z.number(),
    distance: z.number(),
  })
  .passthrough();
export const zDataQuality = z
  .object({
    n_laps_used: z.number(),
    n_laps_rejected: z.number(),
    imputed_features: z.array(z.string()).optional(),
  })
  .passthrough();
export const zUnknownFingerprint = z
  .object({
    fingerprint_id: z.string(),
    fingerprint_vector: z.array(z.number()).nullish(),
    radar_values: radar.optional(),
    style_description: z.string().optional(),
    nearest_neighbours: z.array(zNeighbour).optional(),
    risk_flags: z.array(z.string()).optional(),
    data_quality: zDataQuality,
    confidence_state: z.string().nullish(),
    warnings: z.array(z.string()).optional(),
  })
  .passthrough();

// POST /ingest/session -> 202 (F14). Counts are honest 0-placeholders on enqueue.
export const zIngestSession = z
  .object({
    session_id: z.string(),
    laps_accepted: z.number(),
    laps_rejected: z.number(),
    rejection_summary: z.record(z.number()).optional(),
    quality_report: z.record(z.unknown()).optional(),
  })
  .passthrough();

// GET /ingest/status/{id} (F14). status drives the pipeline rail; result populated on COMPLETE.
export const zIngestStatus = z
  .object({
    status: z.enum(['PENDING', 'PROCESSING', 'COMPLETE', 'FAILED']),
    progress: z.number(),
    error_detail: z.string().nullish(),
    result: z.record(z.unknown()).nullish(),
  })
  .passthrough();

// POST /ingest/upload -> 503 stub (F05, div #58). Guard exists so the report table is a 1-line swap.
export const zUpload = z
  .object({
    upload_id: z.string(),
    laps_accepted: z.number(),
    laps_rejected: z.number(),
    cleaning_report: z.record(z.unknown()).optional(),
    warnings: z.array(z.string()).optional(),
  })
  .passthrough();

// --- F19 Screen 6 (Historical Race Replay Monitor) ------------------------------------------------
// GET /replay/{session}/{driver} (F14 → F10). The route forwards only 5 fields per lap; τ and the
// whole DegradationAlert are dropped (div #72–#75). Guard the load-bearing fields, passthrough rest.
// `laps` is optional so a valid empty replay ({laps: []}) never trips a false UI_SCHEMA_MISMATCH.
export const zReplayLap = z
  .object({
    lap_number: z.number(),
    feature_vector: z.record(z.union([z.number(), z.null()])).optional(),
    deviation_score: z.number().nullish(),
    alert_state: z.string(),
    fingerprint_snapshot: z.record(z.unknown()).nullish(),
  })
  .passthrough();
export const zReplay = z
  .object({
    laps: z.array(zReplayLap).optional(),
  })
  .passthrough();
