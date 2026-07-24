import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { StyleNode, TeammateLink } from '@/lib/style-space/styleSpace';

import { StyleSpace2D } from './StyleSpace2D';

const nodes: StyleNode[] = [
  { code: 'VER', coords2d: [0, 0], coords5d: [1, 0, 0], team: 'Red Bull' },
  { code: 'PER', coords2d: [1, 1], coords5d: [0.9, 0.1, 0], team: 'Red Bull' },
  { code: 'HAM', coords2d: [2, 0], coords5d: [0, 1, 0], team: null }, // absent from /drivers → neutral
];
const teammates: TeammateLink[] = [{ a: 'VER', b: 'PER', similarity: 0.99 }];

const base = {
  nodes,
  teammates: [] as TeammateLink[],
  showTeammates: false,
  selected: null,
  onSelect: () => {},
  onHover: () => {},
};

describe('StyleSpace2D (master §10.6, div #68)', () => {
  it('renders one node per driver', () => {
    render(<StyleSpace2D {...base} />);
    expect(screen.getAllByTestId('style-node')).toHaveLength(3);
  });

  it('colours nodes by team; a null-team node is neutral (div #65)', () => {
    render(<StyleSpace2D {...base} />);
    expect(document.querySelector('[data-driver="VER"]')).toHaveAttribute('fill', '#3671C6'); // Red Bull
    expect(document.querySelector('[data-driver="HAM"]')).toHaveAttribute('fill', '#2B2B2B'); // neutral
  });

  it('shows teammate lines only when enabled', () => {
    const { rerender } = render(<StyleSpace2D {...base} teammates={teammates} showTeammates={false} />);
    expect(screen.queryByTestId('teammate-line')).toBeNull();
    rerender(<StyleSpace2D {...base} teammates={teammates} showTeammates />);
    expect(screen.getByTestId('teammate-line')).toBeInTheDocument();
  });

  it('fires onHover and onSelect for a node', () => {
    const onHover = vi.fn();
    const onSelect = vi.fn();
    render(<StyleSpace2D {...base} onHover={onHover} onSelect={onSelect} />);
    const ver = document.querySelector('[data-driver="VER"]')!;
    fireEvent.mouseEnter(ver);
    expect(onHover).toHaveBeenCalledWith('VER');
    fireEvent.click(ver);
    expect(onSelect).toHaveBeenCalledWith('VER');
  });
});
