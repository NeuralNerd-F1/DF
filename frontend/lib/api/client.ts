import type { ZodTypeAny } from 'zod';

import { logClientError } from '@/lib/log';

import { messageFor, UiErrorCode, type AnyErrorCode, type ApiResult } from './errors';
import type { ErrorCode, ErrorResponse } from './types';

const DEFAULT_TIMEOUT_MS = 10_000;

function baseUrl(): string {
  const u = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8001';
  return u.replace(/\/+$/, '');
}

function readFlags(body: unknown): ErrorCode[] {
  if (body && typeof body === 'object' && 'flags' in body) {
    const f = (body as { flags?: unknown }).flags;
    if (Array.isArray(f) && f.every((x) => typeof x === 'string')) return f as ErrorCode[];
  }
  return [];
}

// §6 "client log" column: server 5xx + the client-side codes are logged; an honest 404/NOT_FOUND
// is not (it is expected absence, not a failure).
function shouldLog(code: AnyErrorCode, status: number): boolean {
  if (code === 'NOT_FOUND') return false;
  if (status >= 500) return true;
  return (
    code === 'DB_UNAVAILABLE' ||
    code === 'HEALTH_FAIL' ||
    code === UiErrorCode.NETWORK ||
    code === UiErrorCode.TIMEOUT ||
    code === UiErrorCode.SCHEMA_MISMATCH
  );
}

/**
 * The one typed fetch boundary. NEVER throws into a component and never returns raw `any`
 * (F16 spec §4.0-a / §3): every path resolves to a typed ApiResult. The abort timer stays armed
 * through the response-BODY read (via `finally`), so a server that stalls the body can't hang the
 * caller — it aborts to UI_TIMEOUT. `schema` runtime-validates a 2xx body; a mismatch →
 * UI_SCHEMA_MISMATCH (no malformed render). Loggable failures are surfaced to the client log.
 */
export async function fetchApi<T>(
  path: string,
  schema: ZodTypeAny,
  init?: RequestInit & { timeoutMs?: number },
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutMs = init?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    let res: Response;
    try {
      res = await fetch(`${baseUrl()}${path}`, {
        ...init,
        signal: controller.signal,
        headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
      });
    } catch (e) {
      const aborted = e instanceof DOMException && e.name === 'AbortError';
      const code: AnyErrorCode = aborted ? UiErrorCode.TIMEOUT : UiErrorCode.NETWORK;
      logClientError(code, path, 'fetch');
      return { ok: false, code, detail: messageFor(code), status: 0 };
    }

    if (!res.ok) {
      let code: AnyErrorCode = res.status === 404 ? 'NOT_FOUND' : 'API_ERROR';
      let detail = '';
      try {
        const body = (await res.json()) as Partial<ErrorResponse> & { detail?: unknown };
        if (typeof body?.error_code === 'string') code = body.error_code as ErrorCode;
        else if (res.status === 422) code = 'INVALID_INPUT'; // Pydantic {detail:[...]} shape
        if (typeof body?.detail === 'string') detail = body.detail;
      } catch {
        /* non-JSON / stalled error body — fall back to the status-derived code/message */
      }
      if (shouldLog(code, res.status)) logClientError(code, detail || path, 'http');
      return { ok: false, code, detail: messageFor(code, detail), status: res.status };
    }

    let json: unknown;
    try {
      json = await res.json();
    } catch {
      // A stalled body aborts here → treat as a timeout, not a schema mismatch.
      const code = controller.signal.aborted ? UiErrorCode.TIMEOUT : UiErrorCode.SCHEMA_MISMATCH;
      logClientError(code, path, 'body');
      return { ok: false, code, detail: messageFor(code), status: res.status };
    }

    const parsed = schema.safeParse(json);
    if (!parsed.success) {
      logClientError(UiErrorCode.SCHEMA_MISMATCH, path, 'validate');
      return {
        ok: false,
        code: UiErrorCode.SCHEMA_MISMATCH,
        detail: messageFor(UiErrorCode.SCHEMA_MISMATCH),
        status: res.status,
      };
    }

    return {
      ok: true,
      data: parsed.data as T,
      status: res.status,
      partial: res.status === 206,
      flags: readFlags(json),
    };
  } finally {
    clearTimeout(timer);
  }
}
