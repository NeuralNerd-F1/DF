import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ postUnknownFingerprint: vi.fn() }));

import { postUnknownFingerprint } from '@/lib/api/endpoints';

import { UnknownProjectionPanel } from './UnknownProjectionPanel';

const ok = (data: unknown) => ({ ok: true, data, status: 200, partial: false, flags: [] });
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const PLACED = {
  fingerprint_id: 'f1',
  fingerprint_vector: [0.1, 0.2, 0.3, 0.4, 0.5],
  data_quality: { n_laps_used: 6, n_laps_rejected: 0 },
  nearest_neighbours: [{ driver: 'VER', cosine_similarity: 0.82, distance: 0.18 }],
  risk_flags: [],
  warnings: ['SPARSE_PROJECTION'],
  confidence_state: 'LOW_CONFIDENCE',
};

afterEach(() => vi.clearAllMocks());

describe('UnknownProjectionPanel (master §10.6, div #67)', () => {
  it('projects → nearest matches + sparse warning, and lifts the 5D vector to the parent', async () => {
    const onProjected = vi.fn();
    (postUnknownFingerprint as Mock).mockResolvedValue(ok(PLACED));
    render(<UnknownProjectionPanel onProjected={onProjected} />);
    fireEvent.click(screen.getByTestId('unknown-project-submit'));
    await waitFor(() => expect(screen.getByTestId('unknown-projection-result')).toBeInTheDocument());
    expect(screen.getByTestId('nearest-neighbours')).toBeInTheDocument();
    expect(screen.getByTestId('unknown-warning')).toHaveAttribute('data-code', 'SPARSE_PROJECTION');
    expect(onProjected).toHaveBeenCalledWith(expect.objectContaining({ fingerprint_vector: [0.1, 0.2, 0.3, 0.4, 0.5] }));
    expect(screen.queryByTestId('unknown-unplaced')).toBeNull();
  });

  it('null fingerprint_vector → unplaced note, no fabricated position (div #67)', async () => {
    (postUnknownFingerprint as Mock).mockResolvedValue(ok({ ...PLACED, fingerprint_vector: null }));
    render(<UnknownProjectionPanel onProjected={() => {}} />);
    fireEvent.click(screen.getByTestId('unknown-project-submit'));
    await waitFor(() => expect(screen.getByTestId('unknown-unplaced')).toBeInTheDocument());
  });

  it('400 INVALID_INPUT → ErrorCard', async () => {
    (postUnknownFingerprint as Mock).mockResolvedValue(err('INVALID_INPUT', 400, 'need >= 5 laps'));
    render(<UnknownProjectionPanel onProjected={() => {}} />);
    fireEvent.click(screen.getByTestId('unknown-project-submit'));
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('INVALID_INPUT'));
  });
});
