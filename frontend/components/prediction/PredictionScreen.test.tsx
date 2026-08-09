import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { PredictionScreen } from './PredictionScreen';

const api = vi.hoisted(() => ({
  getSegments: vi.fn(),
  getDrivers: vi.fn(),
  getSetup: vi.fn(),
  postPredict: vi.fn(),
}));
vi.mock('@/lib/api/endpoints', () => api);
// PageError (the error path) calls useRouter — mock the router context (PageError.test pattern).
vi.mock('next/navigation', () => ({ useRouter: () => ({ refresh: vi.fn() }) }));

const SEGMENT = {
  id: 'Belgium-2022-NOR-38',
  label: 'Belgium 2022 · Race · Lap 38',
  circuit: 'Belgium',
  season: 2022,
  session: 'Race',
  lap_number: 38,
  true_driver: 'NOR',
  features: { F01: 62.1, F02: null },
};

const DRIVERS = [
  { id: 'NOR', code: 'NOR', team: 'McLaren', full_name: 'Lando Norris', radar_values: {} },
];

const PREDICTION = {
  predicted_driver: 'NOR',
  confidence: 0.904,
  confidence_state: 'HIGH',
  top_3: [{ driver: 'NOR', probability: 0.904 }],
  class_probabilities: { NOR: 0.904, ALO: 0.084, STR: 0.012 },
  shap_values: { F04: 2.65, F28: -0.38 },
  shap_available: true,
  model_version: 'v1',
  confidence_breakdown: {
    feature_quality: 0.83,
    distribution: 0.86,
    model_confidence: 0.904,
    overall: 0.83 * 0.86 * 0.904,
  },
};

const SETUP = {
  driver: 'NOR',
  circuit: 'Belgium',
  parameters: [],
  consistency_warning: false,
  monaco_override: false,
  approximation_disclaimer: 'Approximated.',
  model_version: 'v1',
};

beforeEach(() => {
  vi.resetAllMocks();
  api.getSegments.mockResolvedValue({ ok: true, data: [SEGMENT], flags: [] });
  api.getDrivers.mockResolvedValue({ ok: true, data: DRIVERS, flags: [] });
  api.postPredict.mockResolvedValue({ ok: true, data: PREDICTION, flags: [] });
  api.getSetup.mockResolvedValue({ ok: true, data: SETUP, flags: [] });
});

async function pickTheSegment() {
  render(<PredictionScreen />);
  const button = await screen.findByTestId(`segment-${SEGMENT.id}`);
  await userEvent.click(button);
  await waitFor(() => expect(screen.getByTestId('predicted-driver-card')).toBeInTheDocument());
}

describe('PredictionScreen', () => {
  it('predicts on segment pick and grades against the ground truth', async () => {
    await pickTheSegment();
    expect(api.postPredict).toHaveBeenCalledWith({
      feature_vector: SEGMENT.features,
      circuit: 'Belgium',
    });
    expect(screen.getByTestId('predicted-driver-card')).toHaveTextContent('Lando Norris');
    expect(screen.getByTestId('truth-line')).toHaveAttribute('data-correct', 'true');
    expect(screen.getByTestId('truth-line')).toHaveTextContent('correct ✓');
  });

  it('renders the propagated-confidence breakdown with the served numbers', async () => {
    await pickTheSegment();
    expect(screen.getByTestId('confidence-breakdown')).toBeInTheDocument();
    expect(screen.getByTestId('breakdown-data')).toHaveTextContent('0.83');
    expect(screen.getByTestId('breakdown-distribution')).toHaveTextContent('0.86');
    expect(screen.getByTestId('breakdown-model')).toHaveTextContent('0.90');
    expect(screen.getByTestId('breakdown-overall')).toHaveTextContent('0.65');
  });

  it('omits the breakdown when the payload lacks it (older cache) — no fabricated numbers', async () => {
    api.postPredict.mockResolvedValue({
      ok: true,
      data: { ...PREDICTION, confidence_breakdown: null },
      flags: [],
    });
    await pickTheSegment();
    expect(screen.queryByTestId('confidence-breakdown')).toBeNull();
  });

  it('renders the per-lap signed SHAP with both directions', async () => {
    await pickTheSegment();
    expect(screen.getByTestId('signed-bar-F04')).toHaveAttribute('data-direction', 'confirms');
    expect(screen.getByTestId('signed-bar-F28')).toHaveAttribute('data-direction', 'argues');
  });

  it('renders the full class distribution with the predicted row emphasised', async () => {
    await pickTheSegment();
    expect(screen.getByTestId('prob-row-NOR')).toHaveAttribute('data-predicted');
    expect(screen.getByTestId('prob-row-ALO')).not.toHaveAttribute('data-predicted');
    expect(screen.getByTestId('prob-row-NOR')).toHaveTextContent('90.4%');
  });

  it('fetches setup for the PREDICTED driver at the segment circuit', async () => {
    await pickTheSegment();
    await waitFor(() => expect(api.getSetup).toHaveBeenCalledWith('NOR', 'Belgium'));
  });

  it('a gated null prediction renders the uncertainty state, never a forced name', async () => {
    api.postPredict.mockResolvedValue({
      ok: true,
      data: {
        ...PREDICTION,
        predicted_driver: null,
        confidence: 0.31,
        confidence_state: 'LOW_CONFIDENCE',
        shap_values: null,
        shap_available: false,
      },
      flags: [],
    });
    await pickTheSegment();
    expect(screen.getByTestId('gated-prediction')).toHaveTextContent('LOW_CONFIDENCE');
    expect(screen.getByTestId('setup-absent')).toBeInTheDocument();
    expect(api.getSetup).not.toHaveBeenCalled();
  });

  it('an incorrect prediction is graded incorrect — the screen never hides a miss', async () => {
    api.postPredict.mockResolvedValue({
      ok: true,
      data: { ...PREDICTION, predicted_driver: 'ALO' },
      flags: [],
    });
    api.getDrivers.mockResolvedValue({ ok: true, data: [], flags: [] });
    await pickTheSegment();
    expect(screen.getByTestId('truth-line')).toHaveAttribute('data-correct', 'false');
    expect(screen.getByTestId('truth-line')).toHaveTextContent('incorrect ✗');
  });

  it('empty segments → explicit build hint, no picker', async () => {
    api.getSegments.mockResolvedValue({ ok: true, data: [], flags: [] });
    render(<PredictionScreen />);
    expect(await screen.findByTestId('no-segments')).toBeInTheDocument();
    expect(screen.queryByTestId('segment-picker')).toBeNull();
  });

  it('a failed prediction surfaces the error code, not a silent empty state', async () => {
    api.postPredict.mockResolvedValue({ ok: false, code: 'INFERENCE_ERROR', detail: 'down', flags: [] });
    render(<PredictionScreen />);
    await userEvent.click(await screen.findByTestId(`segment-${SEGMENT.id}`));
    await waitFor(() => expect(screen.getByText(/INFERENCE_ERROR/)).toBeInTheDocument());
  });
});
