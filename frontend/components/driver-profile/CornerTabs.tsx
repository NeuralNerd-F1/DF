'use client';

import { useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import { Tabs } from '@/components/ds/Tabs';

const CORNERS = ['Slow', 'Medium', 'Fast'] as const;
type Corner = (typeof CORNERS)[number];

// Corner-type sub-fingerprint tabs (master §10.3). F09 computes these but ships them as an artefact,
// NOT an API endpoint (F16 spec div #49) — so the tabs are built + interactive, but each shows an
// explicit unavailable state, never a fabricated sub-fingerprint. A one-line swap once served.
export function CornerTabs() {
  const [corner, setCorner] = useState<Corner>('Slow');
  return (
    <section data-testid="corner-tabs">
      <h3 className="mb-2 font-mono text-xs text-text-secondary">Corner type</h3>
      <Tabs options={CORNERS} value={corner} onChange={setCorner} ariaLabel="Corner type" />
      <div className="mt-2">
        <EmptyState message={`${corner}-corner sub-fingerprint not yet available.`} />
      </div>
    </section>
  );
}
