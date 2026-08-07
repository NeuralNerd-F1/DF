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
  IngestSessionRequest,
  IngestSessionResponse,
  IngestStatusResponse,
  PredictRequest,
  PredictResponse,
  ReplayResponse,
  SetupResponse,
  StabilityResponse,
  StyleSpaceResponse,
  UnknownFingerprintRequest,
  UnknownFingerprintResponse,
  UploadResponse,
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

export function getFingerprint(code: string): Promise<ApiResult<FingerprintResponse>> {
  return fetchApi<FingerprintResponse>(`/drivers/${enc(code)}/fingerprint`, v.zFingerprint);
}

export function getStability(code: string): Promise<ApiResult<StabilityResponse>> {
  return fetchApi<StabilityResponse>(`/drivers/${enc(code)}/stability`, v.zStability);
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

// --- F17 Screen 3: prediction + ingestion (all POST except getIngestStatus) -----------------------

export function getSegments(): Promise<ApiResult<SampleSegment[]>> {
  return fetchApi<SampleSegment[]>('/segments', v.zSegmentList);
}

export function getTyres(): Promise<ApiResult<TyreWearResponse>> {
  return fetchApi<TyreWearResponse>('/tyres', v.zTyreWear);
}

export function postPredict(body: PredictRequest): Promise<ApiResult<PredictResponse>> {
  return fetchApi<PredictResponse>('/predict', v.zPredict, jsonPost(body));
}

export function postUnknownFingerprint(
  body: UnknownFingerprintRequest,
): Promise<ApiResult<UnknownFingerprintResponse>> {
  return fetchApi<UnknownFingerprintResponse>('/fingerprint/unknown', v.zUnknownFingerprint, jsonPost(body));
}

export function postIngestSession(
  body: IngestSessionRequest,
): Promise<ApiResult<IngestSessionResponse>> {
  return fetchApi<IngestSessionResponse>('/ingest/session', v.zIngestSession, jsonPost(body));
}

export function getIngestStatus(sessionId: string): Promise<ApiResult<IngestStatusResponse>> {
  return fetchApi<IngestStatusResponse>(`/ingest/status/${enc(sessionId)}`, v.zIngestStatus);
}

// Manual upload: multipart file POST. Currently a 503 stub (F05, div #58) — the client surfaces the
// honest error; NO Content-Type header (the browser sets the multipart boundary for FormData).
export function postUpload(file: File): Promise<ApiResult<UploadResponse>> {
  const form = new FormData();
  form.append('file', file);
  return fetchApi<UploadResponse>('/ingest/upload', v.zUpload, { method: 'POST', body: form });
}

// --- F19 Screen 6: historical race replay (F14 → F10) ---------------------------------------------
// Whole-payload pull: the entire replay (all laps) at once; playback is client-side (div #71). The
// session/driver are DB UUIDs — no session catalog endpoint exists (div #77). A fresh checkout 503s
// (no data) until F20 populates the DB; the screen renders that honestly.
export function getReplay(sessionId: string, driverId: string): Promise<ApiResult<ReplayResponse>> {
  return fetchApi<ReplayResponse>(`/replay/${enc(sessionId)}/${enc(driverId)}`, v.zReplay);
}
