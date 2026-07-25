// Convenient aliases over the GENERATED OpenAPI types (lib/api/schema.d.ts). Components import
// from here, never hand-typing an endpoint shape or an error-code string (F16 spec §4.0-c).
import type { components } from './schema';

export type ErrorCode = components['schemas']['ErrorCode'];
export type ConfidenceState = components['schemas']['ConfidenceState'];
export type DriverSummary = components['schemas']['DriverSummary'];
export type FingerprintResponse = components['schemas']['FingerprintResponse'];
export type StabilityResponse = components['schemas']['StabilityResponse'];
export type SetupResponse = components['schemas']['SetupResponse'];
export type SetupParameterSchema = components['schemas']['SetupParameterSchema'];
export type CompareResponse = components['schemas']['CompareResponse'];
export type StyleSpaceResponse = components['schemas']['StyleSpaceResponse'];
export type HealthResponse = components['schemas']['HealthResponse'];

// F17 — Screen 3 (Ingestion & Prediction) + Screen 4 reuse. Aliased from the SAME generated
// components; no backend change (div #53). The snapshot already carries these (F13/F14 defined them).
export type PredictRequest = components['schemas']['PredictRequest'];
export type PredictResponse = components['schemas']['PredictResponse'];
export type TopK = components['schemas']['TopK'];
export type UnknownFingerprintRequest = components['schemas']['UnknownFingerprintRequest'];
export type UnknownFingerprintResponse = components['schemas']['UnknownFingerprintResponse'];
export type NeighbourSchema = components['schemas']['NeighbourSchema'];
export type DataQualitySchema = components['schemas']['DataQualitySchema'];
export type IngestSessionRequest = components['schemas']['IngestSessionRequest'];
export type IngestSessionResponse = components['schemas']['IngestSessionResponse'];
export type IngestStatusResponse = components['schemas']['IngestStatusResponse'];
export type UploadResponse = components['schemas']['UploadResponse'];

// The uniform error envelope (app/api/schemas.py ErrorResponse). Defined explicitly because the
// FastAPI exception handlers don't declare it as a response_model, so it isn't in the OpenAPI
// components — the pytest drift guard asserts the real shape still matches this.
export type ErrorResponse = { error_code: ErrorCode; detail: string; flags?: ErrorCode[] };

// The six SHAP/radar domains are structural (master §6), not config — the frontend receives them
// as radar_values / shap_radar keys and must not re-derive them.
export const RADAR_DOMAINS = [
  'throttle',
  'braking',
  'speed',
  'gear',
  'engine',
  'cornering',
] as const;
export type RadarDomain = (typeof RADAR_DOMAINS)[number];
