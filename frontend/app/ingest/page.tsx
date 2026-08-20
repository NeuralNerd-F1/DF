import type { Metadata } from 'next';

import { PredictionScreen } from '@/components/prediction/PredictionScreen';

export const metadata: Metadata = {
  title: 'APEX Prediction and Setup Tendencies',
  description:
    'Pick a real held-out lap and watch the classifier identify the driver, explain its decision, and estimate setup tendencies.',
};

// Screen 3 — Prediction + Setup Tendencies. The curated held-out-lap prediction experience IS the
// screen: segment → prediction → SHAP explanation → setup tendencies. The ingestion workbench
// (FastF1 job / upload / unknown driver / raw vector) lives on master only — outside this branch's
// presented surface.
export default function IngestPage() {
  return (
    <main className="mx-auto max-w-7xl p-6">
      <PredictionScreen />
    </main>
  );
}
