import type { ZodTypeAny } from 'zod';

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

/**
 * The one typed fetch boundary. NEVER throws into a component and never returns raw `any`
 * (F16 spec §4.0-a / §3): every path resolves to a typed ApiResult. `schema` runtime-validates a
 * 2xx body; a mismatch → UI_SCHEMA_MISMATCH (no malformed render). The compile-time type is `T`
 * (a generated type), bridged by the zod validation.
 */
export async function fetchApi<T>(
  path: string,
  schema: ZodTypeAny,
  init?: RequestInit & { timeoutMs?: number },
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutMs = init?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch(`${baseUrl()}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    });
  } catch (e) {
    clearTimeout(timer);
    const aborted = e instanceof DOMException && e.name === 'AbortError';
    const code: AnyErrorCode = aborted ? UiErrorCode.TIMEOUT : UiErrorCode.NETWORK;
    return { ok: false, code, detail: messageFor(code), status: 0 };
  }
  clearTimeout(timer);

  if (!res.ok) {
    let code: AnyErrorCode = res.status === 404 ? 'NOT_FOUND' : 'API_ERROR';
    let detail = '';
    try {
      const body = (await res.json()) as Partial<ErrorResponse> & { detail?: unknown };
      if (typeof body?.error_code === 'string') code = body.error_code as ErrorCode;
      else if (res.status === 422) code = 'INVALID_INPUT'; // Pydantic {detail:[...]} shape
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      /* non-JSON error body — fall back to the status-derived code/message */
    }
    return { ok: false, code, detail: messageFor(code, detail), status: res.status };
  }

  let json: unknown;
  try {
    json = await res.json();
  } catch {
    const code = UiErrorCode.SCHEMA_MISMATCH;
    return { ok: false, code, detail: messageFor(code), status: res.status };
  }

  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    const code = UiErrorCode.SCHEMA_MISMATCH;
    return { ok: false, code, detail: messageFor(code), status: res.status };
  }

  return {
    ok: true,
    data: parsed.data as T,
    status: res.status,
    partial: res.status === 206,
    flags: readFlags(json),
  };
}
