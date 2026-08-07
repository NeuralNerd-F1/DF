import Link from 'next/link';

import type { DriverSummary } from '@/lib/api/types';
import { driverNumber } from '@/lib/design/drivers';
import { teamColour } from '@/lib/design/teams';

// Screen 2 masthead: race number in team colour (SVG fill — no inline style), full name in the
// display face, team + code tag. Falls back to the code wherever identity data is absent.
export function ProfileHeader({ driver }: { driver: DriverSummary }) {
  const colour = teamColour(driver.team);
  const number = driverNumber(driver.code);

  return (
    <header className="mb-5" data-testid="profile-header">
      <Link href="/" className="font-mono text-[11px] uppercase tracking-[0.15em] text-accent hover:underline">
        ← Back to overview
      </Link>
      <div className="mt-2 flex items-end gap-4">
        <svg
          viewBox={`0 0 ${number.length * 34 + 4} 52`}
          width={number.length * 34 + 4}
          height={52}
          aria-hidden="true"
          className="block shrink-0"
        >
          <text x={0} y={45} fill={colour} className="font-display text-[56px] font-bold">
            {number}
          </text>
        </svg>
        <div className="min-w-0 pb-1">
          <h1 className="truncate font-display text-[32px] font-bold uppercase leading-none tracking-[0.02em] text-text-primary">
            {driver.full_name ?? driver.code}
          </h1>
          <p className="mt-1.5 flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.14em]">
            <svg width={8} height={8} aria-hidden="true" className="inline-block">
              <rect width={8} height={8} fill={colour} />
            </svg>
            <span className="text-text-primary">{driver.team ?? 'Team unknown'}</span>
            <span className="text-text-secondary">#{driver.code}</span>
          </p>
        </div>
      </div>
    </header>
  );
}
