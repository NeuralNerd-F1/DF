import { PageError } from '@/components/ds/PageError';
import { OverviewGrid } from '@/components/overview/OverviewGrid';
import { getDrivers } from '@/lib/api/endpoints';

// SSR the initial driver grid (master §9 — SSR for initial fingerprint data, CSR for animation).
export const dynamic = 'force-dynamic';

export default async function DriverOverviewPage() {
  const res = await getDrivers();
  return (
    <main className="mx-auto max-w-7xl p-6">
      {res.ok ? (
        <OverviewGrid initialDrivers={res.data} />
      ) : (
        <PageError code={res.code} detail={res.detail} />
      )}
    </main>
  );
}
