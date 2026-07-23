import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  getIngestStatus,
  postIngestSession,
  postPredict,
  postUnknownFingerprint,
  postUpload,
} from './endpoints';
import { UiErrorCode } from './errors';

function resp(status: number, body: unknown): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => body } as unknown as Response;
}

type InitCapture = { method?: string; headers?: Record<string, string>; body?: unknown };

const PREDICT_OK = {
  predicted_driver: 'VER',
  confidence: 0.9,
  confidence_state: 'HIGH',
  model_version: 'v1',
  shap_available: true,
  shap_values: { F01: 0.3 },
  top_3: [{ driver: 'VER', probability: 0.9 }],
};

afterEach(() => vi.unstubAllGlobals());

describe('F17 endpoint helpers (Screen 3)', () => {
  it('postPredict POSTs JSON with Content-Type and parses the response', async () => {
    const f = vi.fn().mockResolvedValue(resp(200, PREDICT_OK));
    vi.stubGlobal('fetch', f);
    const r = await postPredict({ feature_vector: { F01: 0.5 }, circuit: 'Bahrain' });
    expect(r.ok).toBe(true);
    const init = f.mock.calls[0]?.[1] as InitCapture;
    expect(String(f.mock.calls[0]?.[0])).toContain('/predict');
    expect(init.method).toBe('POST');
    expect(init.headers).toMatchObject({ 'Content-Type': 'application/json' });
    expect(JSON.parse(String(init.body))).toMatchObject({ circuit: 'Bahrain' });
  });

  it('surfaces a 400 INVALID_INPUT from /predict', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(400, { error_code: 'INVALID_INPUT', detail: 'too many NULLs' })));
    const r = await postPredict({ feature_vector: {}, circuit: 'Bahrain' });
    expect(!r.ok && r.code).toBe('INVALID_INPUT');
  });

  it('surfaces a 503 INFERENCE_ERROR from /predict', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(503, { error_code: 'INFERENCE_ERROR', detail: 'boom' })));
    const r = await postPredict({ feature_vector: { F01: 1 }, circuit: 'Bahrain' });
    expect(!r.ok && r.code).toBe('INFERENCE_ERROR');
  });

  it('postPredict returns UI_SCHEMA_MISMATCH on a malformed body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(200, { confidence: 'nope' })));
    const r = await postPredict({ feature_vector: { F01: 1 }, circuit: 'Bahrain' });
    expect(!r.ok && r.code).toBe(UiErrorCode.SCHEMA_MISMATCH);
  });

  it('postPredict returns UI_NETWORK when fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('down')));
    const r = await postPredict({ feature_vector: { F01: 1 }, circuit: 'Bahrain' });
    expect(!r.ok && r.code).toBe(UiErrorCode.NETWORK);
  });

  it('postUpload posts FormData (no JSON Content-Type) and surfaces the F05 503 stub', async () => {
    const f = vi.fn().mockResolvedValue(resp(503, { error_code: 'HEALTH_FAIL', detail: 'upload ships in F05' }));
    vi.stubGlobal('fetch', f);
    const file = new File(['a,b\n1,2'], 't.csv', { type: 'text/csv' });
    const r = await postUpload(file);
    expect(!r.ok && r.code).toBe('HEALTH_FAIL');
    const init = f.mock.calls[0]?.[1] as InitCapture;
    expect(init.method).toBe('POST');
    expect(init.headers?.['Content-Type']).toBeUndefined();
  });

  it('getIngestStatus parses a terminal COMPLETE status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(200, { status: 'COMPLETE', progress: 1, result: { laps_accepted: 10 } })));
    const r = await getIngestStatus('sess-1');
    expect(r.ok && r.data.status).toBe('COMPLETE');
  });

  it('postIngestSession parses the 202 enqueue response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(resp(202, { session_id: 's1', laps_accepted: 0, laps_rejected: 0 })));
    const r = await postIngestSession({ year: 2024, event: 'Bahrain', session: 'R' });
    expect(r.ok && r.data.session_id).toBe('s1');
  });

  it('postUnknownFingerprint parses nearest-neighbours + warnings', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        resp(200, {
          fingerprint_id: 'f',
          fingerprint_vector: [0, 0, 0, 0, 0],
          data_quality: { n_laps_used: 6, n_laps_rejected: 0 },
          nearest_neighbours: [{ driver: 'VER', cosine_similarity: 0.8, distance: 0.2 }],
          risk_flags: [],
          warnings: ['SPARSE_PROJECTION'],
          confidence_state: 'LOW_CONFIDENCE',
        }),
      ),
    );
    const r = await postUnknownFingerprint({ laps: [{ F01: 1 }], circuit: 'Bahrain' });
    expect(r.ok && r.data.warnings).toContain('SPARSE_PROJECTION');
  });
});
