import type { SetupParameterSchema, SetupResponse } from '@/lib/api/types';
import { FEATURE_LABELS } from '@/lib/design/features';
import { tendencyText } from '@/lib/design/setup';

// Rating → tone. HIGH/MEDIUM ride the confidence tokens; LOW and WIDE both mean "treat with
// caution" and share the low tone. The chip shows the rating WORD — never a re-thresholded number.
const RATING_CLASS: Record<string, string> = {
  HIGH: 'text-confidence-high border-confidence-high/40',
  MEDIUM: 'text-confidence-medium border-confidence-medium/40',
  LOW: 'text-confidence-low border-confidence-low/40',
  WIDE: 'text-confidence-low border-confidence-low/40',
};

function prettyName(name: string): string {
  const s = name.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// Display value per parameter kind, preserving the SetupPanel honesty rules it replaces:
// a CATEGORY shows its label (never the raw ordinal — a number would read as a fake measurement),
// a WIDE parameter shows its range (a point value would overstate the model's certainty), and a
// parameter with no value shows a dash.
function displayValue(p: SetupParameterSchema): string {
  if (String(p.kind) === 'CATEGORY' && p.category) return p.category;
  if (p.value_range && p.value == null) {
    const [lo, hi] = p.value_range;
    return `${lo.toFixed(1)}–${hi.toFixed(1)} ${p.unit}`;
  }
  if (p.value == null) return '—';
  if (typeof p.value === 'string') return p.value;
  const digits = String(p.kind) === 'INDEX' || String(p.kind) === 'INTEGER_LAPS' ? 0 : 1;
  return `${p.value.toFixed(digits)} ${p.unit}`;
}

// The 10-parameter recommendation as compact rows — name, value, rating chip. Response-level
// warnings (consistency, Monaco override) and per-parameter constraint flags surface as chips
// above the disclaimer; they are never dropped (failsafe HIGH-1).
export function SetupList({ setup }: { setup: SetupResponse }) {
  const paramFlags = Array.from(new Set(setup.parameters.flatMap((p) => p.constraint_flags ?? [])));
  const banners: string[] = [];
  if (setup.consistency_warning) banners.push('CONSISTENCY_WARNING');
  if (setup.monaco_override) banners.push('CIRCUIT_OVERRIDE_MONACO');
  const allFlags = Array.from(new Set([...banners, ...paramFlags.map(String)]));

  return (
    <div data-testid="setup-list">
      <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-text-secondary">
        {setup.circuit}
      </p>
      <ul className="mt-2 divide-y divide-border">
        {setup.parameters.map((p) => {
          const tendency = tendencyText(p);
          // System input #8: the honest per-parameter explanation — the design-time basis of the
          // target heuristic (a documented constant), never a per-driver causal claim.
          const basis = (p.basis_features ?? []).map((f) => FEATURE_LABELS[f] ?? f).join(', ');
          return (
            <li
              key={p.name}
              className="py-1.5"
              data-testid={`setup-param-${p.name}`}
              title={`confidence ${p.confidence.toFixed(2)}${
                basis ? ` · target heuristic reads: ${basis}` : ''
              }`}
            >
              <span className="flex items-center justify-between gap-2">
                <span className="min-w-0 truncate font-mono text-[10px] text-text-secondary">
                  {prettyName(p.name)}
                </span>
                <span className="flex shrink-0 items-center gap-2">
                  <span className="font-mono text-[11px] tabular-nums text-text-primary">
                    {displayValue(p)}
                  </span>
                  <span
                    className={`rounded border px-1 py-px font-mono text-[8px] uppercase tracking-[0.1em] ${
                      RATING_CLASS[String(p.rating)] ?? 'border-border text-text-secondary'
                    }`}
                  >
                    {String(p.rating)}
                  </span>
                </span>
              </span>
              {tendency ? (
                <span
                  className="mt-0.5 block font-mono text-[9px] leading-snug text-text-secondary"
                  data-testid={`setup-tendency-${p.name}`}
                >
                  {tendency}
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
      {allFlags.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1" data-testid="setup-flags">
          {allFlags.map((f) => (
            <span
              key={f}
              className="rounded border border-border px-1.5 py-0.5 font-mono text-[9px] text-accent-deep"
            >
              {f}
            </span>
          ))}
        </div>
      ) : null}
      <p className="mt-2 text-[10px] italic leading-snug text-text-secondary">
        {setup.approximation_disclaimer}
      </p>
    </div>
  );
}
