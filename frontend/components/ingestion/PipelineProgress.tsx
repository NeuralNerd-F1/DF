'use client';

// Pipeline-progress animation (master §10.4). Div #55: the async ingestion job exposes only a
// coarse `status` enum + a single `progress` float (no per-stage timings), so the five named stages
// light up proportionally to real progress and a FAILED job turns the reached stage red with the
// job's `error_detail`. We deliberately DO NOT fabricate per-stage green/red timings the API never
// reports.
export type PipelinePhase = 'IDLE' | 'PENDING' | 'PROCESSING' | 'COMPLETE' | 'FAILED';

const STAGES = ['Fetching', 'Validating', 'Extracting Features', 'Running Model', 'Computing SHAP'] as const;

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

export function PipelineProgress({
  status,
  progress,
  errorDetail,
  className = '',
}: {
  status: PipelinePhase;
  progress: number;
  errorDetail?: string | null;
  className?: string;
}) {
  const p = clamp01(progress);
  const done = status === 'COMPLETE' ? STAGES.length : Math.round(p * STAGES.length);
  const failedStage = status === 'FAILED' ? Math.min(STAGES.length - 1, Math.floor(p * STAGES.length)) : -1;

  return (
    <div data-testid="pipeline" data-status={status} className={className}>
      <div className="flex gap-1">
        {STAGES.map((s, i) => {
          const state =
            status === 'FAILED' && i === failedStage
              ? 'failed'
              : i < done
                ? 'done'
                : status !== 'IDLE' && status !== 'COMPLETE' && i === done
                  ? 'active'
                  : 'pending';
          const bar =
            state === 'done'
              ? 'bg-confidence-high'
              : state === 'failed'
                ? 'bg-accent'
                : state === 'active'
                  ? 'bg-confidence-medium animate-pulse'
                  : 'bg-border';
          return (
            <div key={s} data-testid="pipeline-stage" data-stage={s} data-state={state} className="flex-1">
              <div className={`h-1.5 rounded ${bar}`} />
              <span className="mt-1 block truncate font-mono text-[9px] text-text-secondary">{s}</span>
              {state === 'failed' ? (
                <code data-testid="pipeline-error" className="mt-0.5 block font-mono text-[10px] text-accent">
                  {errorDetail ?? 'FAILED'}
                </code>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
