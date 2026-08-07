'use client';

import { useEffect, useMemo, useState } from 'react';

import { SignedBarList } from '@/components/charts/SignedBarList';
import { PageError } from '@/components/ds/PageError';
import { Skeleton } from '@/components/ds/Skeleton';
import { SetupList } from '@/components/driver-profile/SetupList';
import type { AnyErrorCode } from '@/lib/api/errors';
import type {
  DriverSummary,
  PredictResponse,
  SampleSegment,
  SetupResponse,
} from '@/lib/api/types';
import { getDrivers, getSegments, getSetup, postPredict } from '@/lib/api/endpoints';

import { ClassProbabilityChart } from './ClassProbabilityChart';
import { PredictedDriverCard } from './PredictedDriverCard';
import { SegmentPicker } from './SegmentPicker';

type PredictionState =
  | { kind: 'idle' }
  | { kind: 'loading'; segment: SampleSegment }
  | { kind: 'error'; segment: SampleSegment; code: AnyErrorCode; detail?: string }
  | { kind: 'done'; segment: SampleSegment; prediction: PredictResponse };

type SetupState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'absent' }
  | { kind: 'error'; code: AnyErrorCode; detail?: string }
  | { kind: 'done'; setup: SetupResponse };

// Screen 3 hero — pick a real held-out lap, watch the classifier identify the driver, read why.
// Every panel is the model's real output for THAT lap: gated confidence, per-lap signed SHAP, the
// full class distribution, and the setup recommendation for the predicted driver at the segment's
// circuit. The truth line grades the answer against the matrix's known driver.
export function PredictionScreen() {
  const [segments, setSegments] = useState<SampleSegment[] | null>(null);
  const [segmentsError, setSegmentsError] = useState<{ code: AnyErrorCode; detail?: string } | null>(
    null,
  );
  const [drivers, setDrivers] = useState<DriverSummary[]>([]);
  const [state, setState] = useState<PredictionState>({ kind: 'idle' });
  const [setupState, setSetupState] = useState<SetupState>({ kind: 'idle' });

  useEffect(() => {
    let alive = true;
    void getSegments().then((r) => {
      if (!alive) return;
      if (r.ok) setSegments(r.data);
      else setSegmentsError({ code: r.code, detail: r.detail });
    });
    void getDrivers().then((r) => {
      if (alive && r.ok) setDrivers(r.data);
    });
    return () => {
      alive = false;
    };
  }, []);

  const identities = useMemo(() => new Map(drivers.map((d) => [d.code, d])), [drivers]);

  function pick(segment: SampleSegment): void {
    setState({ kind: 'loading', segment });
    setSetupState({ kind: 'idle' });
    void postPredict({ feature_vector: segment.features ?? {}, circuit: segment.circuit }).then(
      (r) => {
        setState((current) => {
          // A stale response for a segment the user has already navigated away from is dropped.
          if (current.kind !== 'loading' || current.segment.id !== segment.id) return current;
          if (!r.ok) return { kind: 'error', segment, code: r.code, detail: r.detail };
          const predicted = r.data.predicted_driver;
          if (predicted) {
            setSetupState({ kind: 'loading' });
            void getSetup(predicted, segment.circuit).then((s) => {
              setSetupState(
                s.ok
                  ? { kind: 'done', setup: s.data }
                  : s.code === 'NOT_FOUND'
                    ? { kind: 'absent' }
                    : { kind: 'error', code: s.code, detail: s.detail },
              );
            });
          } else {
            setSetupState({ kind: 'absent' });
          }
          return { kind: 'done', segment, prediction: r.data };
        });
      },
    );
  }

  const active = state.kind !== 'idle' ? state.segment : null;

  return (
    <div data-testid="prediction-screen">
      <header className="mb-5">
        <h1 className="font-display text-[34px] font-bold uppercase leading-none tracking-[0.03em] text-text-primary">
          Prediction + Setup
        </h1>
        <p className="mt-2 text-[13px] text-text-secondary">
          Select a telemetry segment — the classifier identifies the driver and explains its
          decision.
        </p>
      </header>

      {segments === null && segmentsError === null ? (
        <Skeleton className="h-28 w-full" />
      ) : segmentsError !== null ? (
        <PageError code={segmentsError.code} detail={segmentsError.detail} />
      ) : segments !== null && segments.length === 0 ? (
        <p className="font-mono text-[11px] text-text-secondary" data-testid="no-segments">
          No sample segments available — build the driver profiles artefact first.
        </p>
      ) : segments !== null ? (
        <SegmentPicker
          segments={segments}
          selectedId={active?.id ?? null}
          onSelect={pick}
          disabled={state.kind === 'loading'}
        />
      ) : null}

      {state.kind === 'loading' ? (
        <div className="mt-4 grid gap-4 lg:grid-cols-3" data-testid="prediction-loading">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : null}

      {state.kind === 'error' ? (
        <div className="mt-4">
          <PageError code={state.code} detail={state.detail} />
        </div>
      ) : null}

      {state.kind === 'done' ? (
        <>
          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <PredictedDriverCard
              prediction={state.prediction}
              trueDriver={state.segment.true_driver}
              identity={
                state.prediction.predicted_driver
                  ? identities.get(state.prediction.predicted_driver)
                  : undefined
              }
            />

            <section className="rounded-md border border-border bg-card p-4" data-testid="shap-explanation-card">
              <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
                SHAP explanation
              </h2>
              <p className="mb-3 mt-1 font-mono text-[10px] leading-snug text-text-secondary">
                {state.prediction.predicted_driver
                  ? `Why the model chose ${state.prediction.predicted_driver} for this lap.`
                  : 'Feature contributions for this lap.'}
              </p>
              {state.prediction.shap_available && state.prediction.shap_values ? (
                <SignedBarList
                  values={state.prediction.shap_values}
                  confirmsLabel={`Confirms ${state.prediction.predicted_driver ?? 'prediction'}`}
                />
              ) : (
                <p className="font-mono text-[11px] text-text-secondary" data-testid="shap-unavailable">
                  SHAP unavailable for this prediction.
                </p>
              )}
            </section>

            <section className="rounded-md border border-border bg-card p-4" data-testid="setup-suggestion-card">
              <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
                Setup suggestion
              </h2>
              {setupState.kind === 'loading' ? (
                <Skeleton className="h-40 w-full" />
              ) : setupState.kind === 'done' ? (
                <SetupList setup={setupState.setup} />
              ) : setupState.kind === 'error' ? (
                <PageError code={setupState.code} detail={setupState.detail} />
              ) : (
                <p className="font-mono text-[11px] text-text-secondary" data-testid="setup-absent">
                  No setup recommendation — the gate named no driver to recommend for.
                </p>
              )}
            </section>
          </div>

          <section className="mt-4 rounded-md border border-border bg-card p-4">
            <h2 className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
              Classifier output — all drivers
            </h2>
            <ClassProbabilityChart
              probabilities={state.prediction.class_probabilities ?? {}}
              predicted={state.prediction.predicted_driver ?? null}
              identities={identities}
            />
          </section>
        </>
      ) : null}
    </div>
  );
}
