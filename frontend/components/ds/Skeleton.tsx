// Loading state — an animated telemetry-wave SVG (master §10.1). Deliberately NOT a spinner.
export function Skeleton({
  className = '',
  label = 'Loading',
}: {
  className?: string;
  label?: string;
}) {
  return (
    <div
      className={`animate-pulse ${className}`}
      role="status"
      aria-busy="true"
      aria-live="polite"
      data-testid="skeleton"
    >
      <span className="sr-only">{label}…</span>
      <svg
        viewBox="0 0 120 24"
        preserveAspectRatio="none"
        className="h-full min-h-[24px] w-full"
        aria-hidden="true"
      >
        <path
          d="M0 12 Q 10 2 20 12 T 40 12 T 60 12 T 80 12 T 100 12 T 120 12"
          fill="none"
          className="stroke-border"
          strokeWidth={2}
        />
      </svg>
    </div>
  );
}
