'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

// The five top-level screens. `match` is the pathname prefix that lights the tab — the Driver
// Overview tab also owns /drivers/[code], because a profile is reached by clicking an overview
// card, not by a tab of its own.
const TABS = [
  { label: 'Driver Overview', href: '/', match: ['/', '/drivers'] },
  { label: 'Comparison', href: '/compare', match: ['/compare'] },
  { label: 'Prediction + Setup Tendencies', href: '/ingest', match: ['/ingest'] },
  { label: 'Tyre Analysis', href: '/tyres', match: ['/tyres'] },
  { label: 'Replay', href: '/replay', match: ['/replay'] },
] as const;

function isActive(pathname: string, match: readonly string[]): boolean {
  return match.some((m) => (m === '/' ? pathname === '/' : pathname.startsWith(m)));
}

// Header tab strip — wide-tracked mono uppercase, active tab underlined in the accent red.
export function NavTabs() {
  const pathname = usePathname() ?? '/';
  return (
    <nav className="flex items-stretch gap-1 self-stretch" data-testid="nav-tabs">
      {TABS.map((t) => {
        const active = isActive(pathname, t.match);
        return (
          <Link
            key={t.href}
            href={t.href}
            aria-current={active ? 'page' : undefined}
            className={`relative flex items-center px-3 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
              active ? 'text-text-primary' : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {t.label}
            {active ? (
              <span className="absolute inset-x-3 bottom-0 h-0.5 bg-accent" aria-hidden="true" />
            ) : null}
          </Link>
        );
      })}
    </nav>
  );
}
