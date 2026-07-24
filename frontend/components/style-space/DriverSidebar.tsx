'use client';

import { useEffect, useState } from 'react';

import { FingerprintRadar } from '@/components/charts/FingerprintRadar';
import { Button } from '@/components/ds/Button';
import { EmptyState } from '@/components/ds/EmptyState';
import { getFingerprint } from '@/lib/api/endpoints';
import type { StyleNode } from '@/lib/style-space/styleSpace';

// Click sidebar (master §10.6): the selected driver's fingerprint summary. Reuses the joined
// radar_values for the radar and best-effort fetches nl_description via /drivers/{code}/fingerprint
// (a failure degrades silently — the radar + identity are still shown).
export function DriverSidebar({ node, onClose }: { node: StyleNode; onClose: () => void }) {
  const [nl, setNl] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    void getFingerprint(node.code).then((r) => {
      if (alive && r.ok) setNl(r.data.nl_description ?? null);
    });
    return () => {
      alive = false;
    };
  }, [node.code]);

  return (
    <aside data-testid="driver-sidebar" className="w-64 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-lg text-accent" data-testid="sidebar-code">
          {node.code}
        </h2>
        <Button onClick={onClose} data-testid="sidebar-close" aria-label="Close">
          ✕
        </Button>
      </div>
      <p className="font-mono text-xs text-text-secondary">{node.team ?? 'Unknown team'}</p>
      <div className="mt-3 flex justify-center">
        {node.radar ? (
          <FingerprintRadar values={node.radar} size={200} animate={false} />
        ) : (
          <EmptyState message="No fingerprint radar for this driver." className="py-4" />
        )}
      </div>
      {nl ? <p className="mt-3 text-sm text-text-secondary">{nl}</p> : null}
    </aside>
  );
}
