import { SignedBarList } from '@/components/charts/SignedBarList';
import type { DriverSignal } from '@/lib/api/types';
import { signalDetail, signalText } from '@/lib/design/features';

// Screen 2 evidence card — three stacked layers, each from a different measured source:
//
// 1. SHAP signal weights: mean SIGNED SHAP per feature over the driver's sampled laps (beeswarm
//    artefact via /fingerprint), rendered by the shared SignedBarList (red confirms / blue argues).
// 2. Driver summary: the F07 natural-language description (real model output, not copywriting).
// 3. Key signals: the overview card's three top-attribution features, direction from grid rank.
//
// An empty shap_signed renders the list's explicit absence note — a sign is never derived from
// the unsigned radar magnitudes.
export function SignalWeightsCard({
  code,
  shapSigned,
  nlDescription,
  signature,
}: {
  code: string;
  shapSigned: Record<string, number>;
  nlDescription: string;
  signature: DriverSignal[];
}) {
  return (
    <section className="rounded-md border border-border bg-card p-4" data-testid="signal-weights-card">
      <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        SHAP signal weights
      </h2>
      <p className="mb-3 mt-1 font-mono text-[10px] leading-snug text-text-secondary">
        Mean signed feature contribution toward the {code} classification over their sampled laps.
      </p>
      <SignedBarList values={shapSigned} confirmsLabel={`Confirms ${code}`} />

      <h3 className="mt-4 font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        Driver summary
      </h3>
      {nlDescription ? (
        <p className="mt-1.5 text-[13px] leading-relaxed text-text-primary" data-testid="driver-summary">
          {nlDescription}
        </p>
      ) : (
        <p className="mt-1.5 font-mono text-[11px] text-text-secondary">No description available.</p>
      )}

      <h3 className="mt-4 font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        Key signals
      </h3>
      {signature.length > 0 ? (
        <ul className="mt-1.5 space-y-1">
          {signature.map((s) => (
            <li
              key={s.feature}
              className="flex items-center gap-2 font-mono text-[11px] text-text-primary"
              title={signalDetail(s)}
            >
              <svg width={5} height={5} aria-hidden="true" className="shrink-0">
                <rect width={5} height={5} className="fill-accent" />
              </svg>
              {signalText(s)}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-1.5 font-mono text-[11px] text-text-secondary">No attribution available.</p>
      )}
    </section>
  );
}
