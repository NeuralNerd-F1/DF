import Link from 'next/link';

import { ConfidenceBadge } from '@/components/ds/ConfidenceBadge';
import { ConfidenceBar } from '@/components/ds/ConfidenceBar';
import type { ConfidenceState, DriverSummary, PredictResponse } from '@/lib/api/types';
import { driverNumber } from '@/lib/design/drivers';
import { teamColour } from '@/lib/design/teams';

// The classifier's answer for the picked segment. Everything shown is the model's real output —
// confidence and its GATED state (a LOW_CONFIDENCE / AMBIGUOUS lap renders as exactly that), and
// the ground-truth grade against the matrix's true driver, stated as correct or incorrect. A gated
// null prediction renders the uncertainty state instead of forcing a name.
export function PredictedDriverCard({
  prediction,
  trueDriver,
  identity,
}: {
  prediction: PredictResponse;
  trueDriver: string;
  identity: DriverSummary | undefined;
}) {
  const code = prediction.predicted_driver ?? null;
  const colour = teamColour(identity?.team);
  const correct = code !== null && code === trueDriver;

  return (
    <section className="rounded-md border border-border bg-card p-4" data-testid="predicted-driver-card">
      <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        Predicted driver
      </h2>

      {code !== null ? (
        <div className="mt-2 flex items-end gap-3">
          <svg
            viewBox={`0 0 ${driverNumber(code).length * 26 + 4} 40`}
            width={driverNumber(code).length * 26 + 4}
            height={40}
            aria-hidden="true"
            className="block shrink-0"
          >
            <text x={0} y={34} fill={colour} className="font-display text-[42px] font-bold">
              {driverNumber(code)}
            </text>
          </svg>
          <div className="min-w-0 pb-0.5">
            <p className="truncate font-display text-[22px] font-semibold uppercase leading-none text-text-primary">
              {identity?.full_name ?? code}
            </p>
            <p className="mt-1 truncate font-mono text-[10px] uppercase tracking-[0.14em] text-text-secondary">
              {identity?.team ?? 'Team unknown'}
            </p>
          </div>
        </div>
      ) : (
        <p className="mt-2 font-mono text-[12px] text-text-primary" data-testid="gated-prediction">
          No driver named — the confidence gate returned{' '}
          <span className="text-accent">{prediction.confidence_state ?? 'an uncertain state'}</span>{' '}
          rather than forcing an answer.
        </p>
      )}

      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between">
          <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-text-secondary">
            Confidence
          </span>
          <ConfidenceBadge state={prediction.confidence_state as ConfidenceState | null} />
        </div>
        <ConfidenceBar
          confidence={prediction.confidence}
          state={prediction.confidence_state as ConfidenceState | null}
        />
      </div>

      {/* Propagated-confidence composition (system input #6): three served signals + their product.
          Numbers only — tone/colour stay with the gate's confidence_state above; no client-side
          threshold ever decides anything here. */}
      {prediction.confidence_breakdown ? (
        <dl className="mt-2 grid grid-cols-4 gap-2" data-testid="confidence-breakdown">
          {(
            [
              ['Data', prediction.confidence_breakdown.feature_quality, 'completeness vs the NULL ceiling'],
              ['Distribution', prediction.confidence_breakdown.distribution, 'distance from the AMBIGUOUS entropy boundary'],
              ['Model', prediction.confidence_breakdown.model_confidence, 'calibrated max probability'],
              ['Overall', prediction.confidence_breakdown.overall, 'product of the three factors'],
            ] as const
          ).map(([label, v, hint]) => (
            <div key={label} title={hint}>
              <dt className="font-mono text-[8.5px] uppercase tracking-[0.14em] text-text-secondary">
                {label}
              </dt>
              <dd
                className={`font-mono text-[12px] tabular-nums ${
                  label === 'Overall' ? 'text-text-primary' : 'text-text-secondary'
                }`}
                data-testid={`breakdown-${label.toLowerCase()}`}
              >
                {v.toFixed(2)}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}

      <p
        className={`mt-3 border-t border-border pt-2 font-mono text-[11px] ${
          correct ? 'text-confidence-high' : 'text-confidence-low'
        }`}
        data-testid="truth-line"
        data-correct={correct}
      >
        Actual driver: {trueDriver} — {correct ? 'correct ✓' : 'incorrect ✗'}
      </p>

      {code !== null ? (
        <Link
          href={`/drivers/${code}`}
          className="mt-3 block rounded border border-accent px-3 py-1.5 text-center font-mono text-[10px] uppercase tracking-[0.16em] text-accent transition-colors hover:bg-accent/10"
        >
          View full profile →
        </Link>
      ) : null}
    </section>
  );
}
