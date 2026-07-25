'use client';

import { useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { ErrorCard } from '@/components/ds/ErrorCard';
import { Skeleton } from '@/components/ds/Skeleton';
import { postUpload } from '@/lib/api/endpoints';
import type { ApiResult } from '@/lib/api/errors';
import type { UploadResponse } from '@/lib/api/types';

import { DropZone } from './DropZone';

// Manual Upload tab (master §10.4). Div #58: POST /ingest/upload is a 503 stub deferred to F05
// (UPLOAD_001..004 exist in errors.py but are wired to no live route). We build the drag-drop +
// cleaning-report UI, POST the REAL endpoint, and surface the honest 503 — NEVER a fabricated
// cleaning report. When F05 lands, the CleaningReport branch below is a one-line data swap.
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
            <p className="mt-2 font-mono text-[11px] text-text-secondary">
              Expected: the manual-upload pipeline (CSV cleaning report, UPLOAD_001–004) ships in F05.
            </p>
          </div>
        ) : result && result.ok ? (
          <CleaningReport data={result.data} />
        ) : (
          <EmptyState message="Manual CSV upload arrives in F05 — the pipeline is not yet wired." className="py-3" />
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
        <span className="text-confidence-high">Accepted: {data.laps_accepted}</span>
        <span className="text-accent">Rejected: {data.laps_rejected}</span>
      </div>
      {data.warnings && data.warnings.length > 0 ? (
        <ul className="mt-2 space-y-1" data-testid="upload-warnings">
          {data.warnings.map((w, i) => (
            <li key={i} className="font-mono text-[11px] text-confidence-medium">
              {w}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
