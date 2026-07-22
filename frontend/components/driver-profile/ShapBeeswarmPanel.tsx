import { ShapBeeswarm, type BeeswarmLap } from '@/components/charts/ShapBeeswarm';
import { EmptyState } from '@/components/ds/EmptyState';

// The beeswarm is built + animated against a precise per-lap-SHAP contract, but F13 exposes NO
// per-lap SHAP endpoint (only the aggregated shap_radar) — so with no laps it shows an explicit
// unavailable state, never fabricated dots (F16 spec §4.2 / div #48). Wiring it later is a one-line
// data swap once an additive backend seam serves per-lap SHAP.
export function ShapBeeswarmPanel({ laps }: { laps: BeeswarmLap[] }) {
  return (
    <section data-testid="beeswarm-panel">
      <h3 className="mb-2 font-mono text-xs text-text-secondary">Per-lap SHAP</h3>
      {laps.length > 0 ? (
        <ShapBeeswarm laps={laps} />
      ) : (
        <EmptyState message="Per-lap explanations not yet available." />
      )}
    </section>
  );
}
