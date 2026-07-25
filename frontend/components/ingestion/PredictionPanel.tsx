'use client';

import { useState } from 'react';

import { ShapForcePlot } from '@/components/charts/ShapForcePlot';
import { Button } from '@/components/ds/Button';
import { ConfidenceBadge } from '@/components/ds/ConfidenceBadge';
import { ConfidenceBar } from '@/components/ds/ConfidenceBar';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { postPredict } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type { ConfidenceState, PredictResponse } from '@/lib/api/types';
import { CIRCUITS, DEFAULT_CIRCUIT } from '@/lib/design/circuits';

import { SAMPLE_VECTOR_JSON } from './sampleVector';

// The direct /predict flow (master §10.4). Div #54: the master's "FastF1 session -> prediction" is
// not end-to-end wireable (ingestion is async and returns no prediction), so prediction is exposed
// here as a manual {feature_vector, circuit} panel — the one place the app runs the real classifier
// + SHAP. The confidence colour comes from the backend confidence_state (div #46), and the SHAP
// force plot is a partial (signed values only, div #56).
function parseVector(text: string): { vector: Record<string, number>; error: string | null } {
  try {
    const raw = JSON.parse(text) as unknown;
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      return { vector: {}, error: 'Feature vector must be a JSON object of feature-id -> number.' };
    }
    const vector: Record<string, number> = {};
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      if (typeof v === 'number' && Number.isFinite(v)) vector[k] = v;
      else if (v !== null) return { vector: {}, error: `Feature "${k}" must be a number or null.` };
    }
    return { vector, error: null };
  } catch {
    return { vector: {}, error: 'Invalid JSON.' };
  }
}

export function PredictionPanel() {
  const [text, setText] = useState<string>(SAMPLE_VECTOR_JSON);
  const [circuit, setCircuit] = useState<string>(DEFAULT_CIRCUIT);
  const [loading, setLoading] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResult<PredictResponse> | null>(null);
  const [submitted, setSubmitted] = useState<Record<string, number>>({});

  async function run() {
    const { vector, error } = parseVector(text);
    setParseError(error);
    if (error) return;
    setLoading(true);
    setSubmitted(vector);
    const r = await postPredict({ feature_vector: vector, circuit });
    setResult(r);
    setLoading(false);
  }

  return (
    <section data-testid="prediction-panel" aria-label="Prediction">
      <h2 className="mb-2 font-mono text-sm uppercase tracking-wide text-text-secondary">Prediction</h2>
      <label className="mb-1 block font-mono text-[11px] text-text-secondary">
        Feature vector (synthetic example — edit or paste a real 28-feature vector)
      </label>
      <textarea
        data-testid="feature-vector-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        rows={8}
        className="w-full rounded border border-border bg-background p-2 font-mono text-[11px] text-text-primary"
      />
      <div className="mt-2 flex items-center gap-3">
        <select
          data-testid="predict-circuit"
          aria-label="Circuit"
          value={circuit}
          onChange={(e) => setCircuit(e.target.value)}
          className="rounded border border-border bg-background px-2 py-1 font-mono text-xs text-text-primary"
        >
          {CIRCUITS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <Button active onClick={() => void run()} disabled={loading} data-testid="predict-submit">
          Predict
        </Button>
      </div>

      {parseError ? (
        <p data-testid="parse-error" className="mt-2 font-mono text-[11px] text-accent">
          {parseError}
        </p>
      ) : null}

      <div className="mt-4">
        {loading ? (
          <Skeleton className="h-24" label="Running model" />
        ) : result && !result.ok ? (
          <ErrorCard code={result.code} detail={result.detail} onRetry={() => void run()} />
        ) : result && result.ok ? (
          <PredictionResult data={result.data} vector={submitted} />
        ) : null}
      </div>
    </section>
  );
}

function PredictionResult({ data, vector }: { data: PredictResponse; vector: Record<string, number> }) {
  return (
    <div data-testid="prediction-result">
      <div className="flex items-center gap-3">
        <span className="font-mono text-2xl text-text-primary" data-testid="predicted-driver">
          {data.predicted_driver ?? '—'}
        </span>
        <ConfidenceBadge state={(data.confidence_state as ConfidenceState | null) ?? null} />
      </div>
      <ConfidenceBar
        confidence={data.confidence}
        state={(data.confidence_state as ConfidenceState | null) ?? null}
        className="mt-2"
      />
      {data.top_3 && data.top_3.length > 0 ? (
        <ol data-testid="top-3" className="mt-3 space-y-1">
          {data.top_3.map((t) => (
            <li key={t.driver} className="flex justify-between font-mono text-xs text-text-secondary">
              <span>{t.driver}</span>
              <span>{(t.probability * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ol>
      ) : null}
      <div className="mt-4">
        <h3 className="mb-1 font-mono text-[11px] uppercase text-text-secondary">SHAP force</h3>
        <ShapForcePlot
          shapValues={data.shap_values}
          available={Boolean(data.shap_available)}
          featureValues={vector}
        />
      </div>
    </div>
  );
}
