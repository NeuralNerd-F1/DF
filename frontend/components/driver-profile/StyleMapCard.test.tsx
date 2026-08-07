import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { DriverSummary, StyleSpaceResponse } from '@/lib/api/types';

import { StyleMapCard } from './StyleMapCard';

const STYLE: StyleSpaceResponse = {
  drivers: [
    { id: 'VER', code: 'VER', coords_2d: [1.0, 2.0], coords_5d: [0, 0, 0, 0, 0], season: null },
    { id: 'LEC', code: 'LEC', coords_2d: [3.0, 1.0], coords_5d: [0, 0, 0, 0, 0], season: null },
    { id: 'HAM', code: 'HAM', coords_2d: [2.0, 3.0], coords_5d: [0, 0, 0, 0, 0], season: null },
  ],
} as StyleSpaceResponse;

const SUMMARIES = [
  { id: 'VER', code: 'VER', team: 'Red Bull Racing', radar_values: {} },
  { id: 'LEC', code: 'LEC', team: 'Ferrari', radar_values: {} },
  { id: 'HAM', code: 'HAM', team: 'Mercedes', radar_values: {} },
] as DriverSummary[];

describe('StyleMapCard', () => {
  it('plots every driver and highlights exactly the profiled one', () => {
    render(<StyleMapCard style={STYLE} summaries={SUMMARIES} highlight="VER" />);
    for (const code of ['VER', 'LEC', 'HAM']) {
      expect(screen.getByTestId(`style-dot-${code}`)).toBeInTheDocument();
    }
    expect(screen.getByTestId('style-dot-VER')).toHaveAttribute('data-highlight');
    expect(screen.getByTestId('style-dot-LEC')).not.toHaveAttribute('data-highlight');
    expect(screen.getByText('VER highlighted')).toBeInTheDocument();
  });

  it('every dot links to that driver profile', () => {
    render(<StyleMapCard style={STYLE} summaries={SUMMARIES} highlight="VER" />);
    expect(screen.getByLabelText('Open LEC profile')).toHaveAttribute('href', '/drivers/LEC');
  });

  it('is honest about UMAP axes — no invented physical labels', () => {
    render(<StyleMapCard style={STYLE} summaries={SUMMARIES} highlight="VER" />);
    expect(screen.getByText(/carry no physical units/)).toBeInTheDocument();
  });

  it('empty style space → explicit absence, no chart', () => {
    render(
      <StyleMapCard style={{ drivers: [] } as unknown as StyleSpaceResponse} summaries={[]} highlight="VER" />,
    );
    expect(screen.getByText('No style space computed yet.')).toBeInTheDocument();
    expect(screen.queryByTestId('style-map-svg')).toBeNull();
  });

  it('drivers without 2D coordinates are skipped, not plotted at a guessed origin', () => {
    const partial = {
      drivers: [
        { id: 'VER', code: 'VER', coords_2d: [1, 2], coords_5d: [0, 0, 0, 0, 0], season: null },
        { id: 'XXX', code: 'XXX', coords_2d: null, coords_5d: [0, 0, 0, 0, 0], season: null },
      ],
    } as unknown as StyleSpaceResponse;
    render(<StyleMapCard style={partial} summaries={SUMMARIES} highlight="VER" />);
    expect(screen.getByTestId('style-dot-VER')).toBeInTheDocument();
    expect(screen.queryByTestId('style-dot-XXX')).toBeNull();
  });
});
