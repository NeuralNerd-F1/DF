import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ postPredict: vi.fn() }));

import { postPredict } from '@/lib/api/endpoints';

import { PredictionPanel } from './PredictionPanel';

const ok = (data: unknown) => ({ ok: true, data, status: 200, partial: false, flags: [] });
const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

const HIGH = {
  predicted_driver: 'VER',
  confidence: 0.92,
  confidence_state: 'HIGH',
  model_version: 'v1',
  shap_available: true,
  shap_values: { F01: 0.4, F02: -0.2 },
  top_3: [
    { driver: 'VER', probability: 0.92 },
    { driver: 'HAM', probability: 0.05 },
  ],
};

afterEach(() => vi.clearAllMocks());

async function predict() {
  fireEvent.click(screen.getByTestId('predict-submit'));
}

describe('PredictionPanel (master §10.4)', () => {
  it('renders the predicted driver, top-3 and a SHAP force plot on success', async () => {
    (postPredict as Mock).mockResolvedValue(ok(HIGH));
    render(<PredictionPanel />);
    await predict();
    await waitFor(() => expect(screen.getByTestId('prediction-result')).toBeInTheDocument());
    expect(screen.getByTestId('predicted-driver')).toHaveTextContent('VER');
    expect(screen.getByTestId('top-3')).toBeInTheDocument();
    expect(screen.getByTestId('shap-force')).toBeInTheDocument();
    expect(screen.getByTestId('confidence-bar')).toHaveAttribute('data-tone', 'high');
  });

  it('drives the confidence colour from confidence_state and shows the badge on LOW_CONFIDENCE', async () => {
    (postPredict as Mock).mockResolvedValue(ok({ ...HIGH, confidence_state: 'LOW_CONFIDENCE', confidence: 0.4 }));
    render(<PredictionPanel />);
    await predict();
    await waitFor(() => expect(screen.getByTestId('confidence-badge')).toBeInTheDocument());
    expect(screen.getByTestId('confidence-bar')).toHaveAttribute('data-tone', 'low');
    expect(screen.getByTestId('predicted-driver')).toHaveTextContent('VER'); // answer still shown, never hidden
  });

  it('shows an EmptyState (not a blank plot) when SHAP is unavailable, prediction preserved', async () => {
    (postPredict as Mock).mockResolvedValue(ok({ ...HIGH, shap_available: false, shap_values: null }));
    render(<PredictionPanel />);
    await predict();
    await waitFor(() => expect(screen.getByTestId('predicted-driver')).toBeInTheDocument());
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.queryByTestId('shap-force')).toBeNull();
  });

  it('surfaces a 400 INVALID_INPUT as an ErrorCard', async () => {
    (postPredict as Mock).mockResolvedValue(err('INVALID_INPUT', 400, 'too many NULL features'));
    render(<PredictionPanel />);
    await predict();
    await waitFor(() => expect(screen.getByTestId('error-card')).toBeInTheDocument());
    expect(screen.getByTestId('error-code')).toHaveTextContent('INVALID_INPUT');
  });

  it('surfaces a 503 INFERENCE_ERROR as a retryable ErrorCard', async () => {
    (postPredict as Mock).mockResolvedValue(err('INFERENCE_ERROR', 503, 'boom'));
    render(<PredictionPanel />);
    await predict();
    await waitFor(() => expect(screen.getByTestId('error-code')).toHaveTextContent('INFERENCE_ERROR'));
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });
});
