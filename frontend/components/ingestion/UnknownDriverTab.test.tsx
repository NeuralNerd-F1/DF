import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ postUnknownFingerprint: vi.fn() }));

import { postUnknownFingerprint } from '@/lib/api/endpoints';

import { UnknownDriverTab } from './UnknownDriverTab';

const ok = (data: unknown) => ({ ok: true, data, status: 200, partial: false, flags: [] });
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const SPARSE = {
  fingerprint_id: 'f1',
  fingerprint_vector: [0.1, 0.2, 0.3, 0.4, 0.5],
  radar_values: {}, // div #59: route returns empty
  style_description: '',
  nearest_neighbours: [
    { driver: 'VER', cosine_similarity: 0.82, distance: 0.18 },
    { driver: 'HAM', cosine_similarity: 0.61, distance: 0.39 },
  ],
  risk_flags: ['F19'],
  data_quality: { n_laps_used: 6, n_laps_rejected: 0, imputed_features: [] },
  confidence_state: 'LOW_CONFIDENCE',
  warnings: ['SPARSE_PROJECTION'],
};

afterEach(() => vi.clearAllMocks());

describe('UnknownDriverTab (master §10.4, div #59/#60)', () => {
  it('renders NN, risk flags, data-quality, live confidence + sparse warning', async () => {
    (postUnknownFingerprint as Mock).mockResolvedValue(ok(SPARSE));
    render(<UnknownDriverTab />);
    fireEvent.click(screen.getByTestId('unknown-submit'));
    await waitFor(() => expect(screen.getByTestId('unknown-result')).toBeInTheDocument());
    expect(screen.getByTestId('nearest-neighbours')).toBeInTheDocument();
    expect(screen.getByTestId('risk-flags')).toBeInTheDocument();
    expect(screen.getByTestId('data-quality')).toHaveTextContent('6');
    expect(screen.getByTestId('confidence-badge')).toBeInTheDocument();
    expect(screen.getByTestId('unknown-warning')).toHaveAttribute('data-code', 'SPARSE_PROJECTION');
  });

  it('renders the radar + style-space as explicit unavailable states (never a fake radar)', async () => {
    (postUnknownFingerprint as Mock).mockResolvedValue(ok(SPARSE));
    render(<UnknownDriverTab />);
    fireEvent.click(screen.getByTestId('unknown-submit'));
    await waitFor(() => expect(screen.getByTestId('unknown-result')).toBeInTheDocument());
    expect(screen.getByText(/radar not populated by the API/i)).toBeInTheDocument();
    expect(screen.getByText(/Style Space projection \(Screen 5\) arrives in F18/i)).toBeInTheDocument();
    expect(screen.queryByTestId('fingerprint-radar')).toBeNull();
  });

  it('renders the OUT_OF_DISTRIBUTION warning + risk cards against live signal', async () => {
    (postUnknownFingerprint as Mock).mockResolvedValue(
      ok({ ...SPARSE, confidence_state: 'OUT_OF_DISTRIBUTION', warnings: ['OUT_OF_DISTRIBUTION'], risk_flags: ['F19', 'F20'] }),
    );
    render(<UnknownDriverTab />);
    fireEvent.click(screen.getByTestId('unknown-submit'));
    await waitFor(() => expect(screen.getByTestId('unknown-result')).toBeInTheDocument());
    expect(screen.getByTestId('unknown-warning')).toHaveAttribute('data-code', 'OUT_OF_DISTRIBUTION');
    expect(screen.getAllByTestId('risk-flag')).toHaveLength(2);
  });

  it('surfaces a 400 INVALID_INPUT (e.g. < 5 laps) as an ErrorCard', async () => {
    (postUnknownFingerprint as Mock).mockResolvedValue(err('INVALID_INPUT', 400, 'need at least 5 laps'));
    render(<UnknownDriverTab />);
    fireEvent.click(screen.getByTestId('unknown-submit'));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('INVALID_INPUT'));
  });
});
