'use client';

import { useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { postUpload } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type { UploadResponse } from '@/lib/api/types';

import { DropZone } from './DropZone';

// Manual Upload tab (master §10.4). POSTs the live F05 /ingest/upload pipeline and renders the
// real cleaning report; any failure surfaces the honest error — NEVER a fabricated report.
export function ManualUploadTab() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResult<UploadResponse> | null>(null);

  async function upload(file: File) {
    setLoading(true);
    const r = await postUpload(file);
    setResult(r);
    setLoading(false);
  }

  return (
    <div data-testid="manual-upload-tab">
      <DropZone onFile={(f) => void upload(f)} disabled={loading} />

      <div className="mt-3">
        {loading ? (
          <Skeleton className="h-16" label="Reading file" />
        ) : result && !result.ok ? (
          <div data-testid="upload-unavailable">
            <ErrorCard code={result.code} detail={result.detail} />
          </div>
        ) : result && result.ok ? (
          <CleaningReport data={result.data} />
        ) : (
          <EmptyState message="Drop a telemetry CSV — the upload pipeline cleans it and reports every accepted and rejected lap." className="py-3" />
        )}
      </div>
    </div>
  );
}

// Rendered only once F05 makes /ingest/upload live (never on the 503). Kept so the swap is trivial.
function CleaningReport({ data }: { data: UploadResponse }) {
  return (
    <div data-testid="cleaning-report">
      <div className="flex gap-4 font-mono text-xs">
        <span className="text-text-primary">Accepted: {data.laps_accepted}</span>
        <span className="text-alert-ink">Rejected: {data.laps_rejected}</span>
      </div>
      {data.warnings && data.warnings.length > 0 ? (
        <ul className="mt-2 space-y-1" data-testid="upload-warnings">
          {data.warnings.map((w, i) => (
            <li key={i} className="font-mono text-[11px] text-alert-ink">
              {w}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
