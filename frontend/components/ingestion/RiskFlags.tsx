'use client';

// Risk flags (master §10.4 Unknown Driver tab): red warning cards, one per feature the projector
// flagged as outside the known F1 driver distribution. Real data from POST /fingerprint/unknown.
export function RiskFlags({ flags }: { flags: string[] }) {
  if (!flags.length) return null;
  return (
    <div data-testid="risk-flags" className="space-y-2">
      {flags.map((f) => (
        <div
          key={f}
          role="alert"
          data-testid="risk-flag"
          data-feature={f}
          className="rounded border border-l-4 border-border border-l-accent bg-card p-2"
        >
          <code className="font-mono text-xs text-accent">{f}</code>
          <span className="ml-2 text-[11px] text-text-secondary">
            outside known F1 driver distribution
          </span>
        </div>
      ))}
    </div>
  );
}
