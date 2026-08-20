import type { Metadata } from 'next';

import { TyreWearScreen } from '@/components/tyres/TyreWearScreen';

export const metadata: Metadata = {
  title: 'APEX Tyre Analysis',
  description:
    'Fuel-corrected stint degradation per driver and compound, measured from clean race laps.',
};

// Screen 4 — Tyre Wear Analysis.
export default function TyresPage() {
  return (
    <main className="mx-auto max-w-7xl p-6">
      <TyreWearScreen />
    </main>
  );
}
