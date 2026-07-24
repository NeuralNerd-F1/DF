'use client';

import { ConfidenceBadge } from '@/components/ds/ConfidenceBadge';
import type { ReplayLap } from '@/lib/api/types';
import { narrowSnapshot } from '@/lib/replay/replay';

// The per-lap fingerprint radar (master §10.7, div #75). DEFERRED: fingerprint_snapshot carries F08
// 5D/2D coords + confidence_state, NOT the 6-domain radar_values FingerprintRadar needs, and the
// system computes no per-lap 6-domain radar. So this is an explicit deferred state — the per-lap
// confidence_state (which IS real) is surfaced; a radar is NEVER fabricated from 5D coords.
// Named seam: ReplayLap.radar_values.
export function PerLapRadarPanel({ lap }: { lap: ReplayLap | null }) {
  const snap = narrowSnapshot(lap?.fingerprint_snapshot);
  const confidence = snap?.confidence_state ?? null;
  return (
    <div
      data-testid="per-lap-radar"
      className="rounded-lg border border-dashed border-border bg-card p-4 text-center"
    >
      <p className="font-mono text-[10px] uppercase text-text-secondary">Per-lap fingerprint</p>
      <p className="mt-2 text-xs text-text-secondary" data-testid="per-lap-radar-note">
        Per-lap 6-domain radar not exposed (snapshot is 5D coordinates).
      </p>
      {confidence ? (
        <div className="mt-3 flex items-center justify-center gap-2" data-testid="per-lap-confidence">
          <span className="font-mono text-[10px] text-text-secondary">confidence</span>
          <span className="font-mono text-[10px] text-text-primary">{confidence}</span>
          <ConfidenceBadge state={confidence} />
        </div>
      ) : null}
    </div>
  );
}
