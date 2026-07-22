'use client';

import { useEffect } from 'react';

import { ErrorCard } from '@/components/ds/ErrorCard';
import { logClientError } from '@/lib/log';

// App-Router error boundary (master §10.1 error state): a render-time exception anywhere in a
// screen surfaces the APEX error card with a working Retry (reset), never Next.js's default UI.
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    logClientError('API_ERROR', error.message, 'render');
  }, [error]);
  return (
    <main className="mx-auto max-w-6xl p-6">
      <ErrorCard code="API_ERROR" detail="Something went wrong rendering this screen." onRetry={reset} />
    </main>
  );
}
