import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { StyleNode } from '@/lib/style-space/styleSpace';

import { StyleSpaceLegend } from './StyleSpaceLegend';

const nodes: StyleNode[] = [
  { code: 'VER', coords2d: [0, 0], coords5d: [1, 0], team: 'Red Bull' },
  { code: 'HAM', coords2d: [1, 1], coords5d: [0, 1], team: 'Mercedes' },
  { code: 'XXX', coords2d: [2, 2], coords5d: [0, 0], team: null },
];

describe('StyleSpaceLegend (identifies the otherwise-unlabelled dots)', () => {
  it('lists every plotted driver with its count and a teamless fallback', () => {
    render(<StyleSpaceLegend nodes={nodes} selected={null} onHover={vi.fn()} onSelect={vi.fn()} />);
    expect(screen.getByTestId('style-space-legend')).toHaveTextContent('Drivers (3)');
    expect(screen.getByTestId('legend-row-VER')).toHaveTextContent('VER');
    expect(screen.getByTestId('legend-row-XXX')).toHaveTextContent('no team'); // honest, not fabricated
  });

  it('a row selects on click and highlights the matching driver on hover', () => {
    const onSelect = vi.fn();
    const onHover = vi.fn();
    render(<StyleSpaceLegend nodes={nodes} selected="HAM" onHover={onHover} onSelect={onSelect} />);
    fireEvent.click(screen.getByTestId('legend-row-VER'));
    expect(onSelect).toHaveBeenCalledWith('VER');
    fireEvent.mouseEnter(screen.getByTestId('legend-row-HAM'));
    expect(onHover).toHaveBeenCalledWith('HAM');
    expect(screen.getByTestId('legend-row-HAM')).toHaveAttribute('aria-pressed', 'true');
  });
});
