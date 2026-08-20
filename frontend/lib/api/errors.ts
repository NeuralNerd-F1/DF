import type { ErrorCode } from './types';

// Frontend-only codes for failures that occur BEFORE/AROUND the API boundary and therefore have
// no server error_code (F16 spec §3, div #45). Namespaced UI_* so they can never be confused with
// a backend ErrorCode; rendered in the SAME monospace error-state pattern.
export const UiErrorCode = {
  NETWORK: 'UI_NETWORK',
  TIMEOUT: 'UI_TIMEOUT',
  SCHEMA_MISMATCH: 'UI_SCHEMA_MISMATCH',
} as const;
export type UiErrorCode = (typeof UiErrorCode)[keyof typeof UiErrorCode];

export type AnyErrorCode = ErrorCode | UiErrorCode;

// The typed fetch result. A component ALWAYS receives one of these and renders the matching state;
// the client never throws into a component and never returns raw `any` (F16 spec §4.0-a / §3).
export type ApiOk<T> = {
  ok: true;
  data: T;
  status: number;
  partial: boolean; // true on 206 (some seasons / data missing)
  flags: ErrorCode[]; // ErrorResponse.flags carried through on 2xx-with-flags
};
export type ApiErr = {
  ok: false;
  code: AnyErrorCode;
  detail: string;
  status: number; // 0 for client-side (network/timeout/schema) failures
};
export type ApiResult<T> = ApiOk<T> | ApiErr;

// User-facing message per code. Server `detail` is preferred when present; this is the fallback
// and the source for the client-side UI_* codes (master §5/§10.1 messages).
export const ERROR_MESSAGE: Partial<Record<AnyErrorCode, string>> = {
  NOT_FOUND: 'Driver not found.',
  DB_UNAVAILABLE: 'Database temporarily unavailable.',
  HEALTH_FAIL: 'Service temporarily unavailable.',
  API_ERROR: 'Something went wrong. Please retry.',
  INVALID_INPUT: 'Invalid request.',
  LOW_CONFIDENCE:
    'Insufficient telemetry signal for confident identification. Top candidates shown.',
  AMBIGUOUS: 'Input data does not match F1 telemetry patterns.',
  SHAP_UNAVAILABLE: 'Explanation temporarily unavailable.',
  STALE_DATA: 'Showing cached data.',
  // Codes surfaced in-band by /predict and the unknown-driver projection that had no fallback message.
  INFERENCE_ERROR: 'Prediction failed. Please retry.',
  SPARSE_PROJECTION: 'Sparse projection — fewer than 30 clean laps. Result shown with reduced confidence.',
  OUT_OF_DISTRIBUTION: 'Outside the known F1 driver distribution.',
  UI_NETWORK: 'Cannot reach the APEX API.',
  UI_TIMEOUT: 'Request timed out.',
  UI_SCHEMA_MISMATCH: 'Unexpected data format.',
};

export function messageFor(code: AnyErrorCode, detail?: string): string {
  if (detail && detail.trim().length > 0) return detail;
  return ERROR_MESSAGE[code] ?? 'Request failed.';
}

// Codes for which a Retry button is meaningful (transient / server-side). A 404 is not retryable.
const RETRYABLE = new Set<AnyErrorCode>([
  'DB_UNAVAILABLE',
  'HEALTH_FAIL',
  'API_ERROR',
  'INFERENCE_ERROR',
  UiErrorCode.NETWORK,
  UiErrorCode.TIMEOUT,
]);
export function isRetryable(code: AnyErrorCode): boolean {
  return RETRYABLE.has(code);
}
