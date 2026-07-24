import { afterEach, describe, expect, it, vi } from 'vitest';

import { getReplay } from './endpoints';
import { UiErrorCode } from './errors';

function resp(status: number, body: unknown): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => body } as unknown as Response;
}

const REPLAY_OK = {
  laps: [
    {
      lap_number: 1,
      feature_vector: { F01: 0.1 },
      alert_state: 'NOMINAL',
      deviation_score: 0.2,
      fingerprint_snapshot: { coords_5d: [0, 0, 0, 0, 0], coords_2d: [0, 0], confidence_state: 'HIGH' },
    },
  ],
};

afterEach(() => vi.unstubAllGlobals());

describe('F19 getReplay (Screen 6)', () => {
  it('GETs /replay with an encoded session + driver and parses laps', async () => {
    const f = vi.fn().mockResolvedValue(resp(200, REPLAY_OK));
    vi.stubGlobal('fetch', f);
    const r = await getReplay('sess 1', 'drv/1');
    expect(r.ok && r.data.laps?.length).toBe(1);
    expect(String(f.mock.calls[0]?.[0])).toContain('/replay/sess%201/drv%2F1');
  });

  it('marks a 206 response partial with the REPLAY_PARTIAL flag', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(206, { ...REPLAY_OK, flags: ['REPLAY_PARTIAL'] })));
    const r = await getReplay('s', 'd');
    expect(r.ok && r.partial).toBe(true);
    expect(r.ok && r.flags).toContain('REPLAY_PARTIAL');
  });

  it('surfaces a 503 HEALTH_FAIL (fresh checkout — no data loaded)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(resp(503, { error_code: 'HEALTH_FAIL', detail: 'no session data loaded' })),
    );
    const r = await getReplay('s', 'd');
    expect(!r.ok && r.code).toBe('HEALTH_FAIL');
  });

  it('surfaces a 503 DB_UNAVAILABLE (breaker open at runtime)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(503, { error_code: 'DB_UNAVAILABLE' })));
    const r = await getReplay('s', 'd');
    expect(!r.ok && r.code).toBe('DB_UNAVAILABLE');
  });

  it('surfaces a 404 NOT_FOUND (unknown session/driver)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(404, { error_code: 'NOT_FOUND' })));
    const r = await getReplay('s', 'd');
    expect(!r.ok && r.code).toBe('NOT_FOUND');
  });

  it('returns UI_SCHEMA_MISMATCH on a malformed lap', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(200, { laps: [{ alert_state: 5 }] })));
    const r = await getReplay('s', 'd');
    expect(!r.ok && r.code).toBe(UiErrorCode.SCHEMA_MISMATCH);
  });

  it('returns UI_NETWORK when fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('down')));
    const r = await getReplay('s', 'd');
    expect(!r.ok && r.code).toBe(UiErrorCode.NETWORK);
  });

  it('returns UI_TIMEOUT when the request aborts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new DOMException('aborted', 'AbortError')));
    const r = await getReplay('s', 'd');
    expect(!r.ok && r.code).toBe(UiErrorCode.TIMEOUT);
  });
});
