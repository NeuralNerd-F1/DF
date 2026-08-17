'use client';

import { useState } from 'react';

import { Card } from '@/components/ds/Card';
import { Tabs } from '@/components/ds/Tabs';

import { ManualUploadTab } from './ManualUploadTab';
import { PredictionPanel } from './PredictionPanel';
import { UnknownDriverTab } from './UnknownDriverTab';

// The ingestion workbench, hosted below the Prediction hero on /ingest. Three tabs: the CSV
// upload, the unknown-driver projection, and the raw-vector prediction panel (kept for arbitrary
// input — the hero above covers the curated-segment flow). The async FastF1 queue tab lives on
// master only — the queue is outside this branch's presented surface.
type TabKey = 'upload' | 'unknown' | 'vector';
const TAB_KEYS: readonly TabKey[] = ['upload', 'unknown', 'vector'];
const TAB_LABEL: Record<TabKey, string> = {
  upload: 'Manual Upload',
  unknown: 'Unknown Driver',
  vector: 'Manual Vector',
};

export function IngestionScreen() {
  const [tab, setTab] = useState<TabKey>('upload');

  return (
    <div data-testid="ingestion-screen">
      <h2 className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-text-secondary">
        Ingest &amp; upload
      </h2>
      <Card className="p-4">
        <Tabs
          options={TAB_KEYS}
          value={tab}
          onChange={setTab}
          ariaLabel="Ingestion source"
          renderLabel={(k) => TAB_LABEL[k]}
        />
        <div className="mt-4">
          {tab === 'upload' ? (
            <ManualUploadTab />
          ) : tab === 'unknown' ? (
            <UnknownDriverTab />
          ) : (
            <PredictionPanel />
          )}
        </div>
      </Card>
    </div>
  );
}
