import type { Metadata } from 'next';

import { PageError } from '@/components/ds/PageError';
import { StyleSpaceScreen } from '@/components/style-space/StyleSpaceScreen';
import { getDrivers, getStyleSpace } from '@/lib/api/endpoints';

export const metadata: Metadata = {
  title: 'APEX — Style Space',
  description: 'Driver style space (UMAP) — 3D Three.js scatter with a D3 2D fallback.',
};

// Screen 5 (master §10.6). SSR the style space + the drivers join (team colour + mini radar).
export const dynamic = 'force-dynamic';

export default async function StyleSpacePage() {
  const [style, drivers] = await Promise.all([getStyleSpace(), getDrivers()]);

  if (!style.ok) {
    return (
      <main className="mx-auto max-w-6xl p-6">
        <PageError code={style.code} detail={style.detail} />
      </main>
    );
  }

  // A /drivers failure is non-fatal: the scatter still renders from /stylespace, degrading to
  // neutral spheres + no mini radar (the join tolerates missing drivers).
  const driverList = drivers.ok ? drivers.data : [];
  return <StyleSpaceScreen initialStyle={style.data} initialDrivers={driverList} />;
}
