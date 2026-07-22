// Runtime shape-guards for the responses the frontend consumes. Validated at the fetch boundary
// so a malformed body becomes UI_SCHEMA_MISMATCH instead of a silent bad render (F16 spec §4.0-a).
// These guard the load-bearing fields and `.passthrough()` the rest — the compile-time contract is
// the GENERATED types (lib/api/types.ts); a backend rename is caught by the pytest drift guard.
import { z } from 'zod';

const radar = z.record(z.number());

export const zDriverSummary = z
  .object({
    id: z.string(),
    code: z.string(),
    team: z.string().nullish(),
    radar_values: radar.optional(),
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
