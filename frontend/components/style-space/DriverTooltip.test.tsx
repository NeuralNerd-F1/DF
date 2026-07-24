import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { StyleNode } from '@/lib/style-space/styleSpace';

import { DriverTooltip } from './DriverTooltip';

const node: StyleNode = {
  code: 'VER',
  coords2d: [0, 0],
  coords5d: [1, 0, 0],
  team: 'Red Bull',
  radar: { throttle: 0.6, braking: 0.5 },
};

describe('DriverTooltip (master §10.6, div #64/#66)', () => {
  it('shows code, team, deferred season, neighbours and a mini radar', () => {
    render(<DriverTooltip node={node} neighbours={[{ code: 'PER', similarity: 0.99 }]} />);
    expect(screen.getByTestId('tooltip-code')).toHaveTextContent('VER');
    expect(screen.getByTestId('tooltip-season')).toHaveTextContent('—'); // div #64
    expect(screen.getByTestId('tooltip-neighbours')).toHaveTextContent('PER');
    expect(screen.getByTestId('fingerprint-radar')).toBeInTheDocument();
  });

  it('omits the mini radar when the driver has no joined radar (honest)', () => {
    render(<DriverTooltip node={{ ...node, radar: undefined }} neighbours={[]} />);
    expect(screen.queryByTestId('fingerprint-radar')).toBeNull();
  });
});
