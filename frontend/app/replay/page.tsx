import type { Metadata } from 'next';

import { ReplayMonitorScreen } from '@/components/replay/ReplayMonitorScreen';
import { getDrivers } from '@/lib/api/endpoints';

export const metadata: Metadata = {
  title: 'APEX — Race Replay Monitor',
  description: 'Historical race replay — lap-by-lap degradation monitor (Screen 6).',
};

// Screen 6 (master §10.7). SSR only the driver picker; the replay itself is fetched client-side once
// the user enters a session id (no session-catalog endpoint — div #77).
export const dynamic = 'force-dynamic';

export default async function ReplayPage() {
  const drivers = await getDrivers();
  // A /drivers failure is non-fatal here: the driver picker degrades to a UUID input (div #77).
  const driverList = drivers.ok ? drivers.data : [];
  return <ReplayMonitorScreen initialDrivers={driverList} />;
}
