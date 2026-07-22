import { GaugeNeedle } from '@/components/charts/GaugeNeedle';
import type { SetupResponse } from '@/lib/api/types';

// The 10-parameter setup recommendation (master §10.3). Each parameter is a gauge; a WIDE_CONFIDENCE
// parameter renders a range bar (handled inside GaugeNeedle). Constraint flags + the approximation
// disclaimer are surfaced verbatim (F16 spec §4.4) — the recommendation is never presented as measured.
export function SetupPanel({ setup }: { setup: SetupResponse }) {
  const flags = Array.from(
    new Set(setup.parameters.flatMap((p) => p.constraint_flags ?? [])),
  );
  return (
    <section data-testid="setup-panel">
      <h3 className="mb-2 font-mono text-xs text-text-secondary">Setup — {setup.circuit}</h3>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        {setup.parameters.map((p) => (
          <GaugeNeedle
            key={p.name}
            label={p.name}
            value={p.value ?? null}
            range={(p.value_range as [number, number] | null | undefined) ?? null}
            min={p.fia_min ?? 0}
            max={p.fia_max ?? 1}
            unit={p.unit ?? undefined}
            rating={String(p.rating)}
          />
        ))}
      </div>
      {flags.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1" data-testid="setup-flags">
          {flags.map((f) => (
            <span
              key={String(f)}
              className="rounded border border-border px-1.5 py-0.5 font-mono text-[9px] text-accent-deep"
            >
              {String(f)}
            </span>
          ))}
        </div>
      ) : null}
      <p className="mt-2 text-[10px] italic text-text-secondary">{setup.approximation_disclaimer}</p>
    </section>
  );
}
