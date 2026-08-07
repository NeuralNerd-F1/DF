import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { DriverSummary } from '@/lib/api/types';

import { OverviewCard } from './OverviewCard';

const RADAR = { throttle: 1, braking: 0.5, speed: 0.3, gear: 0.2, engine: 0.1, cornering: 0.4 };

const DRIVER: DriverSummary = {
  id: 'VER',
  code: 'VER',
  team: 'Red Bull Racing',
  full_name: 'Max Verstappen',
  radar_values: { throttle: 3.29 },
  signature: [
    { feature: 'F02', attribution: 1.14, percentile: 1.0 },
    { feature: 'F16', attribution: 0.78, percentile: 1.0 },
    { feature: 'F01', attribution: 0.63, percentile: 0.33 },
  ],
  profile: {
    laps_analysed: 4116,
    seasons: [2022, 2023, 2024, 2025],
    n_circuits: 23,
    lap_time_consistency_s: 1.1359,
    identification_f1: 0.5286,
  },
} as DriverSummary;

describe('OverviewCard', () => {
  it('renders number, surname, team and links to the profile', () => {
    render(<OverviewCard driver={DRIVER} radar={RADAR} animate={false} />);
    const card = screen.getByTestId('overview-card-VER');
    expect(card).toHaveAttribute('href', '/drivers/VER');
    expect(screen.getByTestId('driver-number')).toHaveTextContent('1'); // race number, not code
    expect(screen.getByRole('heading')).toHaveTextContent('VERSTAPPEN');
    expect(card).toHaveTextContent('Red Bull Racing');
  });

  it('renders the three key signals as plain-English sentences', () => {
    render(<OverviewCard driver={DRIVER} radar={RADAR} animate={false} />);
    expect(screen.getByText('Strongly modulated throttle')).toBeInTheDocument();
    expect(screen.getByText('Long gear holds')).toBeInTheDocument();
    expect(screen.getByText('Sparing throttle application')).toBeInTheDocument();
  });

  it('renders the measured stats — laps, consistency, ID rate', () => {
    render(<OverviewCard driver={DRIVER} radar={RADAR} animate={false} />);
    const card = screen.getByTestId('overview-card-VER');
    expect(card).toHaveTextContent('4,116');
    expect(card).toHaveTextContent('±1.14s');
    expect(card).toHaveTextContent('53%');
  });

  it('empty signature → explicit "No attribution available", never invented signals', () => {
    render(
      <OverviewCard driver={{ ...DRIVER, signature: [] }} radar={RADAR} animate={false} />,
    );
    expect(screen.getByText('No attribution available')).toBeInTheDocument();
  });

  it('missing profile → dashes for every stat, not zeros', () => {
    render(
      <OverviewCard driver={{ ...DRIVER, profile: null }} radar={RADAR} animate={false} />,
    );
    const card = screen.getByTestId('overview-card-VER');
    expect(card).not.toHaveTextContent('4,116');
    expect(card.textContent?.match(/—/g)?.length).toBe(3); // laps, consistency, ID rate
  });

  it('null consistency renders a dash even when the rest of the profile is present', () => {
    const p = { ...DRIVER.profile!, lap_time_consistency_s: null };
    render(<OverviewCard driver={{ ...DRIVER, profile: p }} radar={RADAR} animate={false} />);
    const card = screen.getByTestId('overview-card-VER');
    expect(card).toHaveTextContent('4,116'); // laps still real
    expect(card).toHaveTextContent('—'); // consistency dashed
  });

  it('unknown driver code falls back to showing the code as its number', () => {
    render(
      <OverviewCard
        driver={{ ...DRIVER, id: 'ZZZ', code: 'ZZZ', full_name: null }}
        radar={RADAR}
        animate={false}
      />,
    );
    expect(screen.getByTestId('driver-number')).toHaveTextContent('ZZZ');
    expect(screen.getByRole('heading')).toHaveTextContent('ZZZ'); // name falls back to code too
  });
});
