import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({
  postIngestSession: vi.fn(),
  getIngestStatus: vi.fn(),
}));

import { getIngestStatus, postIngestSession } from '@/lib/api/endpoints';

import { useIngestionJob } from './useIngestionJob';

const ok = (data: unknown) => ({ ok: true, data, status: 200, partial: false, flags: [] });
const err = (code: string, status: number) => ({ ok: false, code, detail: '', status });

afterEach(() => vi.clearAllMocks());

describe('useIngestionJob (Screen 3 async poll)', () => {
  it('walks PENDING -> PROCESSING -> COMPLETE, then stops polling', async () => {
    (postIngestSession as Mock).mockResolvedValue(ok({ session_id: 's1', laps_accepted: 0, laps_rejected: 0 }));
    (getIngestStatus as Mock)
      .mockResolvedValueOnce(ok({ status: 'PENDING', progress: 0.1 }))
      .mockResolvedValueOnce(ok({ status: 'PROCESSING', progress: 0.5 }))
      .mockResolvedValue(ok({ status: 'COMPLETE', progress: 1, result: { laps_accepted: 10 } }));

    const { result } = renderHook(() => useIngestionJob(10));
    await act(async () => {
      await result.current.submit({ year: 2024, event: 'Bahrain', session: 'R' });
    });
    await waitFor(() => expect(result.current.state.status).toBe('COMPLETE'));
    expect(result.current.state.phase).toBe('complete');
    expect(result.current.state.result).toMatchObject({ laps_accepted: 10 });

    const atComplete = (getIngestStatus as Mock).mock.calls.length;
    await new Promise((r) => setTimeout(r, 60));
    expect((getIngestStatus as Mock).mock.calls.length).toBe(atComplete); // interval cleared on terminal
  });

  it('surfaces a QUEUE_UNAVAILABLE enqueue failure and never polls', async () => {
    (postIngestSession as Mock).mockResolvedValue(err('QUEUE_UNAVAILABLE', 503));
    const { result } = renderHook(() => useIngestionJob(10));
    await act(async () => {
      await result.current.submit({ year: 2024, event: 'Bahrain', session: 'R' });
    });
    expect(result.current.state.phase).toBe('error');
    expect(result.current.state.error?.code).toBe('QUEUE_UNAVAILABLE');
    expect(getIngestStatus as Mock).not.toHaveBeenCalled();
  });

  it('marks the job failed and carries the error_detail', async () => {
    (postIngestSession as Mock).mockResolvedValue(ok({ session_id: 's1', laps_accepted: 0, laps_rejected: 0 }));
    (getIngestStatus as Mock).mockResolvedValue(ok({ status: 'FAILED', progress: 0.4, error_detail: 'INGEST_TASK_FAILED' }));
    const { result } = renderHook(() => useIngestionJob(10));
    await act(async () => {
      await result.current.submit({ year: 2024, event: 'X', session: 'R' });
    });
    await waitFor(() => expect(result.current.state.phase).toBe('failed'));
    expect(result.current.state.errorDetail).toBe('INGEST_TASK_FAILED');
  });
});
