import type { Metadata } from 'next';

import { IngestionScreen } from '@/components/ingestion/IngestionScreen';

export const metadata: Metadata = {
  title: 'APEX — Ingestion & Prediction',
  description: 'Run the classifier, ingest a FastF1 session, upload telemetry, or project an unknown driver.',
};

// Screen 3 (master §10.4). A thin server shell around the interactive client screen.
export default function IngestPage() {
  return <IngestionScreen />;
}
