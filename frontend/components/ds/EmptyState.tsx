// Empty state — a 200 with no data (e.g. GET /drivers -> []). Distinct from an error (master §8.5).
export function EmptyState({ message, className = '' }: { message: string; className?: string }) {
  return (
    <div
      data-testid="empty-state"
      className={`rounded-lg border border-dashed border-border bg-card p-8 text-center text-sm text-text-secondary ${className}`}
    >
      {message}
    </div>
  );
}
