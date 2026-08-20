import { FEATURE_LABELS } from '@/lib/design/features';

// Shared signed-attribution bar list (Screen 2 signal weights, Screen 3 prediction explanation).
// Positive = toward the driver (accent red, the delta.up token), negative = away (blue,
// delta.down) — the force-plot convention. Bars scale to the largest |value| in the list. The
// "argues against" legend renders only when a negative bar is actually on screen, so the legend
// never advertises an encoding not in use.
export function SignedBarList({
  values,
  limit = 6,
  confirmsLabel,
}: {
  values: Record<string, number>;
  limit?: number;
  confirmsLabel: string;
}) {
  const ranked = Object.entries(values)
    .filter(([, v]) => Number.isFinite(v))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]) || a[0].localeCompare(b[0]))
    .slice(0, limit);
  const maxAbs = ranked.length > 0 ? Math.abs(ranked[0][1]) : 0;
  const hasNegative = ranked.some(([, v]) => v < 0);

  if (ranked.length === 0 || maxAbs === 0) {
    return (
      <p className="font-mono text-[11px] text-text-secondary" data-testid="signed-absent">
        No signed attribution available.
      </p>
    );
  }

  return (
    <div data-testid="signed-bar-list">
      <ul className="space-y-1.5">
        {ranked.map(([feature, value]) => {
          const width = Math.max(2, Math.round((Math.abs(value) / maxAbs) * 100));
          const confirms = value >= 0;
          const label = FEATURE_LABELS[feature] ?? feature;
          return (
            <li
              key={feature}
              className="flex items-center gap-2"
              data-testid={`signed-bar-${feature}`}
              data-direction={confirms ? 'confirms' : 'argues'}
              title={`${feature}, signed SHAP ${value.toFixed(3)}`}
            >
              <span className="w-36 shrink-0 truncate text-right font-mono text-[10px] text-text-secondary">
                {label.charAt(0).toUpperCase() + label.slice(1)}
              </span>
              <svg viewBox="0 0 100 6" preserveAspectRatio="none" className="h-2 flex-1" aria-hidden="true">
                <rect width={width} height={6} className={confirms ? 'fill-delta-up' : 'fill-delta-down'} />
              </svg>
              <span
                className={`w-12 shrink-0 font-mono text-[11px] tabular-nums ${
                  confirms ? 'text-delta-up' : 'text-delta-down'
                }`}
              >
                {confirms ? '+' : '−'}
                {Math.abs(value).toFixed(2)}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 flex items-center gap-4 border-t border-border pt-2 font-mono text-[9px] uppercase tracking-[0.14em] text-text-secondary">
        <span className="flex items-center gap-1.5">
          <svg width={10} height={6} aria-hidden="true">
            <rect width={10} height={6} className="fill-delta-up" />
          </svg>
          {confirmsLabel}
        </span>
        {hasNegative ? (
          <span className="flex items-center gap-1.5">
            <svg width={10} height={6} aria-hidden="true">
              <rect width={10} height={6} className="fill-delta-down" />
            </svg>
            Argues against
          </span>
        ) : null}
      </p>
    </div>
  );
}
