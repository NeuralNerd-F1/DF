import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({
  getDrivers: vi.fn(),
  postIngestSession: vi.fn(),
  getIngestStatus: vi.fn(),
}));

import { getDrivers, getIngestStatus, postIngestSession } from '@/lib/api/endpoints';

import { FastF1Tab } from './FastF1Tab';

const ok = (data: unknown) => ({ ok: true, data, status: 200, partial: false, flags: [] });
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const DRIVERS = [
  { id: 'VER', code: 'VER', team: 'Red Bull', radar_values: {} },
  { id: 'HAM', code: 'HAM', team: 'Mercedes', radar_values: {} },
];

afterEach(() => vi.clearAllMocks());

function setEvent(v: string) {
  fireEvent.change(screen.getByTestId('event-input'), { target: { value: v } });
}

describe('FastF1Tab (master §10.4, div #54/#55/#57)', () => {
  it('shows static seasons, drivers from /drivers, and an honest laps-available state', async () => {
    (getDrivers as Mock).mockResolvedValue(ok(DRIVERS));
    render(<FastF1Tab />);
    expect(screen.getByTestId('season-select')).toBeInTheDocument();
    expect(screen.getByTestId('empty-state')).toHaveTextContent(/Laps available: not exposed/i);
    await waitFor(() => expect(screen.getByRole('option', { name: 'VER' })).toBeInTheDocument());
  });

  it('enqueues then polls to COMPLETE and renders the ingestion report', async () => {
    (getDrivers as Mock).mockResolvedValue(ok(DRIVERS));
    (postIngestSession as Mock).mockResolvedValue(ok({ session_id: 's1', laps_accepted: 0, laps_rejected: 0 }));
    (getIngestStatus as Mock).mockResolvedValue(ok({ status: 'COMPLETE', progress: 1, result: { laps_accepted: 12, laps_rejected: 3 } }));
    render(<FastF1Tab />);
    setEvent('Bahrain');
    fireEvent.click(screen.getByTestId('ingest-submit'));
    await waitFor(() => expect(screen.getByTestId('ingest-result')).toBeInTheDocument());
    expect(screen.getByTestId('pipeline')).toHaveAttribute('data-status', 'COMPLETE');
  });

  it('surfaces a QUEUE_UNAVAILABLE enqueue failure without fabricating progress', async () => {
    (getDrivers as Mock).mockResolvedValue(ok(DRIVERS));
    (postIngestSession as Mock).mockResolvedValue(err('QUEUE_UNAVAILABLE', 503, 'broker down'));
    render(<FastF1Tab />);
    setEvent('Bahrain');
    fireEvent.click(screen.getByTestId('ingest-submit'));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('QUEUE_UNAVAILABLE'));
    expect(screen.queryByTestId('ingest-result')).toBeNull();
    expect(getIngestStatus as Mock).not.toHaveBeenCalled();
  });
});
