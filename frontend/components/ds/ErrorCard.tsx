'use client';

import { isRetryable, messageFor, type AnyErrorCode } from '@/lib/api/errors';

// Error state (master §10.1): red left border on the affected card, the error_code in monospace,
// and a Retry button where the call is retryable. This is the browser's no-silent-failure surface.
export function ErrorCard({
  code,
  detail,
  onRetry,
  className = '',
}: {
  code: AnyErrorCode;
  detail?: string;
  onRetry?: () => void;
  className?: string;
}) {
  const showRetry = Boolean(onRetry) && isRetryable(code);
  return (
    <div
      role="alert"
      data-testid="error-card"
      className={`rounded-lg border border-l-4 border-border border-l-accent bg-card p-4 ${className}`}
    >
      <p className="text-sm text-text-primary">{messageFor(code, detail)}</p>
      <code className="mt-1 block font-mono text-xs text-text-secondary" data-testid="error-code">
        {code}
      </code>
      {showRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded border border-border px-3 py-1 font-mono text-xs text-accent transition-colors hover:border-accent"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
