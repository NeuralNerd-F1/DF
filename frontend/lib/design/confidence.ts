import type { ConfidenceState } from '../api/types';

// Confidence colour is driven by the BACKEND-RESOLVED confidence_state, NEVER a client-side
// re-threshold of the numeric confidence (F16 spec §4.0-b, div #46). There is deliberately NO
// 0.65 / 0.85 literal anywhere in the frontend — the backend owns the gate.
export type ConfidenceTone = 'high' | 'medium' | 'low';

export function confidenceTone(state: ConfidenceState | null | undefined): ConfidenceTone {
  switch (state) {
    case 'HIGH':
      return 'high';
    case 'MEDIUM':
      return 'medium';
    case 'LOW_CONFIDENCE':
    case 'AMBIGUOUS':
    case 'OUT_OF_DISTRIBUTION':
      return 'low';
    default:
      return 'low'; // unknown/absent state → treat as low, never as confident
  }
}

// Tailwind classes per tone (the colours themselves live once in tailwind.config.js).
export const CONFIDENCE_TEXT_CLASS: Record<ConfidenceTone, string> = {
  high: 'text-confidence-high',
  medium: 'text-confidence-medium',
  low: 'text-confidence-low',
};
// SVG fill classes — dynamic bar colour set via an SVG attribute + Tailwind class, so no forbidden
// inline `style` is needed for a dynamic colour (F16 spec §4.0-e / §7.1 no-inline-styles).
export const CONFIDENCE_FILL_CLASS: Record<ConfidenceTone, string> = {
  high: 'fill-confidence-high',
  medium: 'fill-confidence-medium',
  low: 'fill-confidence-low',
};

export function isLowConfidence(state: ConfidenceState | null | undefined): boolean {
  return state === 'LOW_CONFIDENCE' || state === 'AMBIGUOUS' || state === 'OUT_OF_DISTRIBUTION';
}
