// Convenient aliases over the GENERATED OpenAPI types (lib/api/schema.d.ts). Components import
// from here, never hand-typing an endpoint shape or an error-code string (F16 spec §4.0-c).
import type { components } from './schema';

export type ErrorCode = components['schemas']['ErrorCode'];
export type ConfidenceState = components['schemas']['ConfidenceState'];
export type DriverSummary = components['schemas']['DriverSummary'];
export type DriverSignal = components['schemas']['DriverSignalSchema'];
export type FingerprintResponse = components['schemas']['FingerprintResponse'];
export type StabilityResponse = components['schemas']['StabilityResponse'];
export type SetupResponse = components['schemas']['SetupResponse'];
export type SetupParameterSchema = components['schemas']['SetupParameterSchema'];
export type CompareResponse = components['schemas']['CompareResponse'];
export type StyleSpaceResponse = components['schemas']['StyleSpaceResponse'];
export type HealthResponse = components['schemas']['HealthResponse'];

// Screen 3 (Prediction + Setup Tendencies) + Screen 4 reuse. Aliased from the SAME generated
// components; no backend change (div #53).
export type SampleSegment = components['schemas']['SampleSegmentSchema'];
export type TyreWearResponse = components['schemas']['TyreWearResponse'];
export type TyreCompoundStats = components['schemas']['TyreCompoundStatsSchema'];
export type PredictRequest = components['schemas']['PredictRequest'];
export type PredictResponse = components['schemas']['PredictResponse'];
export type TopK = components['schemas']['TopK'];

// F19 — Screen 6 (Historical Race Replay Monitor). Aliased from the SAME generated components
// (div #70). The replay redesign widened ReplayLap (threshold + alert — div #72–#74 closed) and
// added the /sessions catalog (div #77 closed).
export type ReplayResponse = components['schemas']['ReplayResponse'];
export type ReplayLap = components['schemas']['ReplayLap'];
export type ReplayAlert = components['schemas']['ReplayAlert'];
export type SessionSummary = components['schemas']['SessionSummary'];

// alert_state is a bare `str` on the wire (StintMonitorState.value, app/monitoring/detector.py).
// The canonical list of the 8 states the monitor emits — the exhaustiveness fixture `alertTone`
// is proven against, so a new monitor state fails the test rather than silently falling through.
export const ALERT_STATES = [
  'INACTIVE',
  'BASELINE',
  'SKIPPED',
  'PENDING',
  'NOMINAL',
  'WATCH',
  'ALERT',
  'HALTED',
] as const;

// fingerprint_snapshot is an open dict on the wire (app/replay/engine.py _snapshot_to_dict). Narrow
// to the three keys it carries; confidence_state is the ONE real per-lap datum — coords are F08
// 5D/2D, NOT a 6-domain radar (div #75).
export type ReplayFingerprintSnapshot = {
  coords_5d?: number[] | null;
  coords_2d?: number[] | null;
  confidence_state?: ConfidenceState | null;
};

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
