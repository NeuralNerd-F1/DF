import type { Metadata } from 'next';

import { IngestionScreen } from '@/components/ingestion/IngestionScreen';
import { PredictionScreen } from '@/components/prediction/PredictionScreen';

export const metadata: Metadata = {
  title: 'APEX — Prediction + Setup Tendencies',
  description:
    'Pick a real held-out lap and watch the classifier identify the driver — or ingest a FastF1 session, upload telemetry, and project an unknown driver.',
};

// Screen 3 — Prediction + Setup. The hero is the curated held-out-lap prediction experience; the
// ingestion workbench (FastF1 job / upload / unknown driver / raw vector) sits below it.
export default function IngestPage() {
  return (
    <main className="mx-auto max-w-7xl p-6">
      <PredictionScreen />
      <div className="mt-8">
        <IngestionScreen />
      </div>
    </main>
  );
}
