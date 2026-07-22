import { DriverGrid } from '@/components/command-centre/DriverGrid';
import { PageError } from '@/components/ds/PageError';
import { getDrivers } from '@/lib/api/endpoints';

// SSR the initial driver grid (master §9 — SSR for initial fingerprint data, CSR for animation).
export const dynamic = 'force-dynamic';

export default async function CommandCentrePage() {
  const res = await getDrivers();
  return (
    <main className="mx-auto max-w-6xl p-6">
      <header className="mb-6">
        <h1 className="font-mono text-2xl text-accent">APEX — Command Centre</h1>
        <p className="text-xs text-text-secondary">Driver behavioural fingerprints</p>
      </header>
      {res.ok ? (
        <DriverGrid initialDrivers={res.data} />
      ) : (
        <PageError code={res.code} detail={res.detail} />
      )}
    </main>
  );
}
