import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { NavTabs } from './NavTabs';

const usePathname = vi.hoisted(() => vi.fn<() => string>());
vi.mock('next/navigation', () => ({ usePathname }));

function activeLabels(): string[] {
  return screen
    .getAllByRole('link')
    .filter((a) => a.getAttribute('aria-current') === 'page')
    .map((a) => a.textContent ?? '');
}

describe('NavTabs', () => {
  beforeEach(() => usePathname.mockReset());

  it('lights exactly the Driver Overview tab on /', () => {
    usePathname.mockReturnValue('/');
    render(<NavTabs />);
    expect(activeLabels()).toEqual(['Driver Overview']);
  });

  it('a driver profile page also belongs to the Driver Overview tab', () => {
    usePathname.mockReturnValue('/drivers/LEC');
    render(<NavTabs />);
    expect(activeLabels()).toEqual(['Driver Overview']);
  });

  it('lights the matching tab for every top-level screen', () => {
    for (const [path, label] of [
      ['/compare', 'Comparison'],
      ['/ingest', 'Prediction + Setup Tendencies'],
      ['/tyres', 'Tyre Analysis'],
      ['/replay', 'Replay'],
    ] as const) {
      usePathname.mockReturnValue(path);
      const { unmount } = render(<NavTabs />);
      expect(activeLabels()).toEqual([label]);
      unmount();
    }
  });

  it('an unknown path lights nothing rather than defaulting to home', () => {
    usePathname.mockReturnValue('/nowhere');
    render(<NavTabs />);
    expect(activeLabels()).toEqual([]);
  });
});
