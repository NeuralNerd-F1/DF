// Typed endpoint calls the screens use. Each returns a typed ApiResult of the GENERATED response
// type; runtime validation is the matching zod guard. The browser NEVER calls /metrics (403).
import { fetchApi } from './client';
import type { ApiResult } from './errors';
import type {
  SampleSegment,
  TyreWearResponse,
  CompareResponse,
  DriverSummary,
  FingerprintResponse,
  HealthResponse,
  PredictRequest,
  PredictResponse,
  ReplayResponse,
  SessionSummary,
  SetupResponse,
  StyleSpaceResponse,
} from './types';
import * as v from './validators';

const enc = encodeURIComponent;

// POST helpers add Content-Type (client.ts only defaults Accept); body is JSON-serialised.
const jsonPost = (body: unknown): RequestInit => ({
  method: 'POST',
  body: JSON.stringify(body),
  headers: { 'Content-Type': 'application/json' },
});

export function getHealth(): Promise<ApiResult<HealthResponse>> {
  return fetchApi<HealthResponse>('/health', v.zHealth);
}

export function getDrivers(): Promise<ApiResult<DriverSummary[]>> {
  return fetchApi<DriverSummary[]>('/drivers', v.zDriverSummaryList);
}

// Cross-season stability rides along on the fingerprint payload (`stability`), so the Driver
// Profile never issues a second call for it — there is deliberately no getStability wrapper.
export function getFingerprint(code: string): Promise<ApiResult<FingerprintResponse>> {
  return fetchApi<FingerprintResponse>(`/drivers/${enc(code)}/fingerprint`, v.zFingerprint);
}

export function getSetup(code: string, circuit: string): Promise<ApiResult<SetupResponse>> {
  return fetchApi<SetupResponse>(`/drivers/${enc(code)}/setup/${enc(circuit)}`, v.zSetup);
}

export function getStyleSpace(): Promise<ApiResult<StyleSpaceResponse>> {
  return fetchApi<StyleSpaceResponse>('/stylespace', v.zStyleSpace);
}

export function getCompare(a: string, b: string): Promise<ApiResult<CompareResponse>> {
  return fetchApi<CompareResponse>(`/compare/${enc(a)}/${enc(b)}`, v.zCompare);
}

// --- Screen 3: prediction over curated held-out segments ------------------------------------------

export function getSegments(): Promise<ApiResult<SampleSegment[]>> {
  return fetchApi<SampleSegment[]>('/segments', v.zSegmentList);
}

export function getTyres(): Promise<ApiResult<TyreWearResponse>> {
  return fetchApi<TyreWearResponse>('/tyres', v.zTyreWear);
}

export function postPredict(body: PredictRequest): Promise<ApiResult<PredictResponse>> {
  return fetchApi<PredictResponse>('/predict', v.zPredict, jsonPost(body));
}

// --- F19 Screen 6: historical race replay (F14 → F10) ---------------------------------------------
// Whole-payload pull: the entire replay (all laps) at once; playback is client-side (div #71). The
// session comes from the /sessions catalog (div #77 closed); the driver is a code from /drivers.
// A fresh checkout 503s (no data) until the DB is populated; the screen renders that honestly.
export function getReplay(sessionId: string, driverId: string): Promise<ApiResult<ReplayResponse>> {
  // The replay is the documented heavy read (26 causal monitor re-runs per request), and grid mode
  // queues several of them against one worker — give it a budget above the 10s default.
  return fetchApi<ReplayResponse>(`/replay/${enc(sessionId)}/${enc(driverId)}`, v.zReplay, {
    timeoutMs: 30_000,
  });
}

// GET /sessions — every persisted session with its replayable driver codes (the session picker).
export function getSessions(): Promise<ApiResult<SessionSummary[]>> {
  return fetchApi<SessionSummary[]>('/sessions', v.zSessions);
}
