import type { ConfidenceState } from '@/lib/api/types';
import { CONFIDENCE_FILL_CLASS, confidenceTone } from '@/lib/design/confidence';

// The confidence gate, rendered. Colour comes from the backend-resolved confidence_state (tone),
// fill fraction from the numeric confidence — NEVER a client-side 0.65/0.85 re-threshold (div #46).
// Dynamic width/colour are SVG attributes + a Tailwind fill class (no inline style).
export function ConfidenceBar({
  confidence,
  state,
  className = '',
}: {
  confidence: number;
  state: ConfidenceState | null | undefined;
  className?: string;
}) {
  const tone = confidenceTone(state);
  const pct = Math.max(0, Math.min(1, confidence)) * 100;
  return (
    <svg
      viewBox="0 0 100 6"
      preserveAspectRatio="none"
      role="img"
      aria-label={`confidence ${tone} ${Math.round(pct)} percent`}
      className={`h-1.5 w-full ${className}`}
      data-testid="confidence-bar"
      data-tone={tone}
    >
      <rect x={0} y={0} width={100} height={6} className="fill-border" />
      <rect
        x={0}
        y={0}
        width={pct}
        height={6}
        className={CONFIDENCE_FILL_CLASS[tone]}
        data-testid="confidence-fill"
      />
    </svg>
  );
}
