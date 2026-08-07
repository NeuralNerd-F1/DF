import type { SampleSegment } from '@/lib/api/types';

// The telemetry-segment picker — real held-out laps (circuits the classifier never trained on).
// The card names circuit / season / session / lap and deliberately NOT the driver: the reveal
// happens after the prediction, so the screen reads as a genuine identification test.
export function SegmentPicker({
  segments,
  selectedId,
  onSelect,
  disabled = false,
}: {
  segments: SampleSegment[];
  selectedId: string | null;
  onSelect: (segment: SampleSegment) => void;
  disabled?: boolean;
}) {
  return (
    <section className="rounded-md border border-border bg-card p-4" data-testid="segment-picker">
      <h2 className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        Telemetry segment
      </h2>
      <p className="mt-1 font-mono text-[10px] leading-snug text-text-secondary">
        Real laps from the held-out circuits — the classifier never saw these circuits in training.
        Pick one and it identifies the driver.
      </p>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
        {segments.map((s) => {
          const selected = s.id === selectedId;
          return (
            <button
              key={s.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelect(s)}
              data-testid={`segment-${s.id}`}
              data-selected={selected || undefined}
              className={`rounded border px-3 py-2 text-left transition-colors focus:outline-none focus:ring-1 focus:ring-accent disabled:opacity-50 ${
                selected
                  ? 'border-accent bg-accent/10'
                  : 'border-border hover:border-text-secondary'
              }`}
            >
              <span
                className={`block font-mono text-[10px] uppercase tracking-[0.12em] ${
                  selected ? 'text-accent' : 'text-text-secondary'
                }`}
              >
                {s.circuit} {s.season}
              </span>
              <span className="mt-0.5 block truncate text-[12px] text-text-primary">
                {s.session || 'Session'} · Lap {s.lap_number}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
