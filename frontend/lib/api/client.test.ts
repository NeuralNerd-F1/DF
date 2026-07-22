import { afterEach, describe, expect, it, vi } from 'vitest';

import { fetchApi } from './client';
import { UiErrorCode } from './errors';
import { zDriverSummaryList } from './validators';

function resp(status: number, body: unknown): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => body } as unknown as Response;
}

const OK_BODY = [{ id: 'VER', code: 'VER', team: 'Red Bull', radar_values: { throttle: 0.5 } }];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchApi failure-mode contract (F16 §6)', () => {
  it('returns ok + data on a valid 200', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(200, OK_BODY)));
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.data).toHaveLength(1);
      expect(r.partial).toBe(false);
    }
  });

  it('flags partial=true on 206', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(206, OK_BODY)));
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(r.ok && r.partial).toBe(true);
  });

  it('maps 404 ErrorResponse to NOT_FOUND', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(resp(404, { error_code: 'NOT_FOUND', detail: 'gone', flags: [] })),
    );
    const r = await fetchApi('/drivers/NOPE/fingerprint', zDriverSummaryList);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.code).toBe('NOT_FOUND');
  });

  it('maps a 503 ErrorResponse to its DB_UNAVAILABLE code', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(resp(503, { error_code: 'DB_UNAVAILABLE', detail: 'db down' })),
    );
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(!r.ok && r.code).toBe('DB_UNAVAILABLE');
  });

  it('falls back to API_ERROR on a 500 with no parseable body', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error('not json');
        },
      } as unknown as Response),
    );
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(!r.ok && r.code).toBe('API_ERROR');
  });

  it('maps 422 to INVALID_INPUT', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(422, { detail: [{ msg: 'bad' }] })));
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(!r.ok && r.code).toBe('INVALID_INPUT');
  });

  it('returns UI_SCHEMA_MISMATCH on a 200 body that fails validation', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(200, [{ code: 123 }])));
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(!r.ok && r.code).toBe(UiErrorCode.SCHEMA_MISMATCH);
  });

  it('returns UI_NETWORK when fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('failed to fetch')));
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(!r.ok && r.code).toBe(UiErrorCode.NETWORK);
    if (!r.ok) expect(r.status).toBe(0);
  });

  it('returns UI_TIMEOUT when the request aborts', async () => {
    const abort = new DOMException('aborted', 'AbortError');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abort));
    const r = await fetchApi('/drivers', zDriverSummaryList);
    expect(!r.ok && r.code).toBe(UiErrorCode.TIMEOUT);
  });

  it('logs client-side failures to the client log (§6 client-log column)', async () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('down')));
    await fetchApi('/drivers', zDriverSummaryList);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('does NOT log an honest 404 (expected absence, not a failure)', async () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(resp(404, { error_code: 'NOT_FOUND', detail: 'x' })),
    );
    await fetchApi('/drivers/X/fingerprint', zDriverSummaryList);
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});
