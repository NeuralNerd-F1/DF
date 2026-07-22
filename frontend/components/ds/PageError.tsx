'use client';

import { useRouter } from 'next/navigation';

import type { AnyErrorCode } from '@/lib/api/errors';

import { ErrorCard } from './ErrorCard';

// Wires a working Retry onto a page-level (SSR) fetch failure: router.refresh() re-runs the Server
// Component, re-issuing the fetch (F16 spec §6 — "Retry button where the call is retryable").
export function PageError({ code, detail }: { code: AnyErrorCode; detail?: string }) {
  const router = useRouter();
  return <ErrorCard code={code} detail={detail} onRetry={() => router.refresh()} />;
}
