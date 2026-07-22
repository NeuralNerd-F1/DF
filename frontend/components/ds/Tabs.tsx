'use client';

// Generic tab strip — used for the season switcher, corner-type tabs, and sort controls.
export function Tabs<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  renderLabel,
  className = '',
}: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  ariaLabel: string;
  renderLabel?: (v: T) => string;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={`inline-flex gap-1 rounded border border-border p-0.5 ${className}`}
    >
      {options.map((o) => (
        <button
          key={o}
          type="button"
          role="tab"
          aria-selected={o === value}
          onClick={() => onChange(o)}
          className={`rounded px-2 py-1 font-mono text-xs transition-colors ${
            o === value ? 'bg-accent text-text-primary' : 'text-text-secondary hover:text-text-primary'
          }`}
        >
          {renderLabel ? renderLabel(o) : o}
        </button>
      ))}
    </div>
  );
}
