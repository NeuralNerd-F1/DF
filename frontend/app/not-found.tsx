import Link from 'next/link';

import { EmptyState } from '@/components/ds/EmptyState';

// App-Router 404 (APEX-styled, not Next's default).
export default function NotFound() {
  return (
    <main className="mx-auto max-w-6xl p-6">
      <EmptyState message="Page not found." />
      <Link href="/" className="mt-4 inline-block font-mono text-xs text-accent hover:underline">
        ← Command Centre
      </Link>
    </main>
  );
}
