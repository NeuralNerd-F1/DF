import type { Metadata } from 'next';

import { ComparisonScreen } from '@/components/comparison/ComparisonScreen';
import { PageError } from '@/components/ds/PageError';
import { getDrivers } from '@/lib/api/endpoints';

export const metadata: Metadata = {
  title: 'APEX Comparison',
  description:
    'Compare two drivers: shared-scale radars, signed SHAP delta, distinctive features, and setup delta.',
};

// Screen 4 (master §10.5). SSR the driver list for the pickers (master §9: SSR data, CSR animation).
export const dynamic = 'force-dynamic';

export default async function ComparePage() {
  const res = await getDrivers();
  if (!res.ok) {
    return (
      <main className="mx-auto max-w-7xl p-6">
        <PageError code={res.code} detail={res.detail} />
      </main>
    );
  }
  return (
    <main className="mx-auto max-w-7xl p-6">
      <ComparisonScreen initialDrivers={res.data} />
    </main>
  );
}
