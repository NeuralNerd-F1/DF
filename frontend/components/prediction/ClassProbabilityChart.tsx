import type { DriverSummary } from '@/lib/api/types';
import { teamColour } from '@/lib/design/teams';

// The full calibrated distribution — every class the model considered, not just the winner. The
// interesting cases are the flat ones: a LOW_CONFIDENCE lap LOOKS flat here, which is the gate's
// justification made visible. Bars are team-coloured; the predicted driver's row is emphasised.
export function ClassProbabilityChart({
  probabilities,
  predicted,
  identities,
}: {
  probabilities: Record<string, number>;
  predicted: string | null;
  identities: Map<string, DriverSummary>;
}) {
  const rows = Object.entries(probabilities)
    .filter(([, p]) => Number.isFinite(p))
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (rows.length === 0) {
    return (
      <p className="font-mono text-[11px] text-text-secondary" data-testid="probabilities-absent">
        No class distribution returned.
      </p>
    );
  }
  const max = Math.max(...rows.map(([, p]) => p), 1e-9);

  return (
    <ul className="space-y-1" data-testid="class-probability-chart">
      {rows.map(([code, p]) => {
        const isPredicted = code === predicted;
        const width = Math.max(0.5, (p / max) * 100);
        const colour = teamColour(identities.get(code)?.team);
        return (
          <li
            key={code}
            className="flex items-center gap-2"
            data-testid={`prob-row-${code}`}
            data-predicted={isPredicted || undefined}
          >
            <span
              className={`w-9 shrink-0 font-mono text-[10px] ${
                isPredicted ? 'text-text-primary' : 'text-text-secondary'
              }`}
            >
              {code}
            </span>
            <svg viewBox="0 0 100 6" preserveAspectRatio="none" className="h-2 flex-1" aria-hidden="true">
              <rect width={100} height={6} className="fill-border/40" />
              <rect width={width} height={6} fill={colour} opacity={isPredicted ? 1 : 0.55} />
            </svg>
            <span
              className={`w-12 shrink-0 text-right font-mono text-[10px] tabular-nums ${
                isPredicted ? 'text-text-primary' : 'text-text-secondary'
              }`}
            >
              {(p * 100).toFixed(1)}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}
