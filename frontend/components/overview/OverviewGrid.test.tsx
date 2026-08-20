import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { DriverSummary } from '@/lib/api/types';

import { OverviewGrid } from './OverviewGrid';

const driver = (code: string, f1: number, laps: number): DriverSummary =>
  ({
    id: code,
    code,
    team: 'Ferrari',
    full_name: null,
    radar_values: { throttle: 1 },
    signature: [{ feature: 'F01', attribution: 0.5, percentile: 1.0 }],
    profile: {
      laps_analysed: laps,
      seasons: [2022, 2025],
      n_circuits: 23,
      lap_time_consistency_s: 1.2,
      identification_f1: f1,
    },
  }) as DriverSummary;

describe('OverviewGrid', () => {
  it('orders cards by ID rate descending and says so', () => {
    render(
      <OverviewGrid
        initialDrivers={[driver('STR', 0.11, 100), driver('LEC', 0.68, 200)]}
        animate={false}
      />,
    );
    const grid = screen.getByTestId('overview-grid');
    const cards = within(grid).getAllByRole('link');
    expect(cards.map((c) => c.getAttribute('href'))).toEqual(['/drivers/LEC', '/drivers/STR']);
    expect(screen.getByText(/ordered by id rate/i)).toBeInTheDocument();
  });

  it('derives the subtitle from the payload — driver count, lap sum, season span', () => {
    render(
      <OverviewGrid
        initialDrivers={[driver('LEC', 0.68, 3914), driver('VER', 0.53, 4116)]}
        animate={false}
      />,
    );
    expect(
      screen.getByText('2 behavioural fingerprints, 8,030 clean laps of telemetry, 2022 to 2025 seasons.'),
    ).toBeInTheDocument();
  });

  it('omits unmeasured subtitle parts instead of printing zeros', () => {
    const bare = { id: 'AAA', code: 'AAA', team: null, radar_values: {} } as DriverSummary;
    render(<OverviewGrid initialDrivers={[bare]} animate={false} />);
    expect(screen.getByText('1 behavioural fingerprint.')).toBeInTheDocument();
  });

  it('empty payload → empty state, no grid', () => {
    render(<OverviewGrid initialDrivers={[]} />);
    expect(screen.queryByTestId('overview-grid')).toBeNull();
    expect(screen.getByText(/no driver fingerprints yet/i)).toBeInTheDocument();
  });

  it('explains the three stats in the legend footnote', () => {
    render(<OverviewGrid initialDrivers={[driver('LEC', 0.68, 100)]} animate={false} />);
    expect(screen.getByText(/held-out F1/)).toBeInTheDocument();
    expect(screen.getByText(/scaled to the grid maximum/)).toBeInTheDocument();
  });
});
