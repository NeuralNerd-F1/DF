import type { ConfidenceState } from '@/lib/api/types';
import { CONFIDENCE_TEXT_CLASS, confidenceTone, isLowConfidence } from '@/lib/design/confidence';

const LABEL: Partial<Record<ConfidenceState, string>> = {
  LOW_CONFIDENCE: 'LOW CONFIDENCE',
  AMBIGUOUS: 'AMBIGUOUS',
  OUT_OF_DISTRIBUTION: 'OUT OF DISTRIBUTION',
};

// The warning badge (master §10.2). Renders ONLY for an uncertainty state — a confident driver
// shows no badge. Never hides the state; surfaces it explicitly.
export function ConfidenceBadge({ state }: { state: ConfidenceState | null | undefined }) {
  if (!state || !isLowConfidence(state)) return null;
  const tone = confidenceTone(state);
  return (
    <span
      role="status"
      data-testid="confidence-badge"
      className={`inline-flex items-center gap-1 rounded border border-current px-1.5 py-0.5 font-mono text-[10px] uppercase leading-none ${CONFIDENCE_TEXT_CLASS[tone]}`}
    >
      <span aria-hidden="true">⚠</span>
      {LABEL[state] ?? state}
    </span>
  );
}
