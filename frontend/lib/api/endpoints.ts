// Typed endpoint calls the screens use. Each returns a typed ApiResult of the GENERATED response
// type; runtime validation is the matching zod guard. The browser NEVER calls /metrics (403).
import { fetchApi } from './client';
import type { ApiResult } from './errors';
import type {
  CompareResponse,
  DriverSummary,
  FingerprintResponse,
  HealthResponse,
  SetupResponse,
  StabilityResponse,
  StyleSpaceResponse,
} from './types';
import * as v from './validators';

const enc = encodeURIComponent;

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
