import { GaugeNeedle } from '@/components/charts/GaugeNeedle';
import type { SetupResponse } from '@/lib/api/types';

// The 10-parameter setup recommendation (master §10.3). Each parameter is a gauge; a WIDE_CONFIDENCE
// parameter renders a range bar. A CATEGORY parameter (suspension) shows its Soft/Medium/Stiff LABEL
// (carried in `category`), never the raw ordinal — a needle+number would read as a fake measurement
// (failsafe HIGH-2). Recommendation-level CONSISTENCY_WARNING / Monaco override live on the response
// (not on a parameter), so they are surfaced as banners too (failsafe HIGH-1) — never dropped.
export function SetupPanel({ setup }: { setup: SetupResponse }) {
  const paramFlags = Array.from(new Set(setup.parameters.flatMap((p) => p.constraint_flags ?? [])));
  const banners: string[] = [];
  if (setup.consistency_warning) banners.push('CONSISTENCY_WARNING');
  if (setup.monaco_override) banners.push('CIRCUIT_OVERRIDE_MONACO');
  const allFlags = Array.from(new Set([...banners, ...paramFlags.map(String)]));

  return (
    <section data-testid="setup-panel">
      <h3 className="mb-2 font-mono text-xs text-text-secondary">Setup — {setup.circuit}</h3>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        {setup.parameters.map((p) => {
          const isCategory = String(p.kind) === 'CATEGORY';
          const displayValue = isCategory && p.category ? p.category : (p.value ?? null);
          return (
            <GaugeNeedle
              key={p.name}
              label={p.name}
              value={displayValue}
              range={(p.value_range as [number, number] | null | undefined) ?? null}
              min={p.fia_min ?? 0}
              max={p.fia_max ?? 1}
              unit={p.unit ?? undefined}
              rating={String(p.rating)}
            />
          );
        })}
      </div>
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
      <p className="mt-2 text-[10px] italic text-text-secondary">{setup.approximation_disclaimer}</p>
    </section>
  );
}
