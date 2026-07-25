'use client';

import { useState } from 'react';

import { Card } from '@/components/ds/Card';
import { Tabs } from '@/components/ds/Tabs';

import { FastF1Tab } from './FastF1Tab';
import { ManualUploadTab } from './ManualUploadTab';
import { PredictionPanel } from './PredictionPanel';
import { UnknownDriverTab } from './UnknownDriverTab';

// Screen 3 — Ingestion & Prediction (master §10.4). Div #54: the master conflates ingestion and
// prediction into one FastF1 flow, but the real API separates them — so a prominent direct
// Prediction panel runs the classifier + SHAP, and the three ingestion tabs below drive the async
// job / upload / unknown-driver projection.
type TabKey = 'fastf1' | 'upload' | 'unknown';
const TAB_KEYS: readonly TabKey[] = ['fastf1', 'upload', 'unknown'];
const TAB_LABEL: Record<TabKey, string> = {
  fastf1: 'FastF1 Session',
  upload: 'Manual Upload',
  unknown: 'Unknown Driver',
};

export function IngestionScreen() {
  const [tab, setTab] = useState<TabKey>('fastf1');

  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="mb-4 font-mono text-lg uppercase tracking-wide text-accent">
        Ingestion &amp; Prediction
      </h1>

      <Card className="mb-6 p-4">
        <PredictionPanel />
      </Card>

      <Card className="p-4">
        <h2 className="mb-3 font-mono text-sm uppercase tracking-wide text-text-secondary">Ingestion</h2>
        <Tabs
          options={TAB_KEYS}
          value={tab}
          onChange={setTab}
          ariaLabel="Ingestion source"
          renderLabel={(k) => TAB_LABEL[k]}
        />
        <div className="mt-4">
          {tab === 'fastf1' ? <FastF1Tab /> : tab === 'upload' ? <ManualUploadTab /> : <UnknownDriverTab />}
        </div>
      </Card>
    </div>
  );
}
