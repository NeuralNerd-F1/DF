import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { DriverSummary, StabilityResponse } from '@/lib/api/types';

import { FactsCard } from './FactsCard';

const DRIVER: DriverSummary = {
  id: 'VER',
  code: 'VER',
  team: 'Red Bull Racing',
  full_name: 'Max Verstappen',
  radar_values: {},
  signature: [],
  profile: {
    laps_analysed: 4116,
    seasons: [2022, 2023, 2024, 2025],
    n_circuits: 23,
    lap_time_consistency_s: 1.1359,
    identification_f1: 0.5286,
  },
} as DriverSummary;

describe('FactsCard', () => {
  it('renders the measured dataset + evaluation numbers', () => {
    render(<FactsCard driver={DRIVER} stability={null} />);
    const card = screen.getByTestId('facts-card');
    expect(card).toHaveTextContent('4,116');
    expect(card).toHaveTextContent('±1.14s');
    expect(card).toHaveTextContent('53%');
    expect(card).toHaveTextContent('2022–2025');
    expect(card).toHaveTextContent('23');
  });

  it('missing profile → dashes everywhere, never zeros', () => {
    render(<FactsCard driver={{ ...DRIVER, profile: null }} stability={null} />);
    expect(screen.getByTestId('facts-card').textContent?.match(/—/g)?.length).toBeGreaterThanOrEqual(5);
  });

  it('a partial stability window says so instead of presenting a score', () => {
    const stab = {
      driver_id: 'VER',
      stability_score: null,
      pairwise_js: {},
      seasons_available: 1,
      partial: true,
    } as StabilityResponse;
    render(<FactsCard driver={DRIVER} stability={stab} />);
    expect(screen.getByTestId('facts-card')).toHaveTextContent('1 season (partial)');
  });

  it('a real stability score renders as the number', () => {
    const stab = {
      driver_id: 'VER',
      stability_score: 0.87,
      pairwise_js: {},
      seasons_available: 3,
      partial: false,
    } as StabilityResponse;
    render(<FactsCard driver={DRIVER} stability={stab} />);
    expect(screen.getByTestId('facts-card')).toHaveTextContent('0.87');
  });
});
