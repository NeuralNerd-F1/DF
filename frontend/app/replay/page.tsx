import type { Metadata } from 'next';

import { ReplayMonitorScreen } from '@/components/replay/ReplayMonitorScreen';
import { getDrivers } from '@/lib/api/endpoints';

export const metadata: Metadata = {
  title: 'APEX Race Replay Monitor',
  description: 'Historical race replay. A lap-by-lap behavioural deviation monitor.',
};

// Screen 6 (master §10.7). SSR only the driver identities (team colours + the catalog-unavailable
// fallback picker); the session catalog and the replay itself are fetched client-side off
// /sessions + /replay (div #77 closed — sessions are picked, not typed).
export const dynamic = 'force-dynamic';

export default async function ReplayPage() {
  const drivers = await getDrivers();
  // A /drivers failure is non-fatal here: team colours degrade to neutral and the fallback picker
  // degrades to a UUID input; the session catalog carries the selection either way.
  const driverList = drivers.ok ? drivers.data : [];
  return (
    <main className="mx-auto max-w-7xl p-6">
      <ReplayMonitorScreen initialDrivers={driverList} />
    </main>
  );
}
