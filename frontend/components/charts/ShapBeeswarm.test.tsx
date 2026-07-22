import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ShapBeeswarm, type BeeswarmLap } from './ShapBeeswarm';

const laps: BeeswarmLap[] = [
  { lapNumber: 1, shap: 0.4, circuit: 'Monaco', session: 'R', confidence: 0.9 },
  { lapNumber: 2, shap: -0.2, circuit: 'Monaco' },
  { lapNumber: 3, shap: 0.1, circuit: 'Spain' },
];

function dotPositions(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll('circle')).map(
    (c) => `${c.getAttribute('cx')},${c.getAttribute('cy')}`,
  );
}

describe('ShapBeeswarm', () => {
  it('renders one dot per lap, coloured by circuit', () => {
    const { container } = render(<ShapBeeswarm laps={laps} />);
    expect(container.querySelectorAll('circle')).toHaveLength(3);
    const monaco = container.querySelector('circle[data-lap="1"]')?.getAttribute('fill');
    const monaco2 = container.querySelector('circle[data-lap="2"]')?.getAttribute('fill');
    const spain = container.querySelector('circle[data-lap="3"]')?.getAttribute('fill');
    expect(monaco).toBe(monaco2); // same circuit → same colour
    expect(spain).not.toBe(monaco); // different circuit → different colour
  });

  it('is deterministic — the same laps render identical dot positions (reproducibility #3)', () => {
    const a = render(<ShapBeeswarm laps={laps} />);
    const posA = dotPositions(a.container);
    a.unmount();
    const b = render(<ShapBeeswarm laps={laps} />);
    expect(dotPositions(b.container)).toEqual(posA);
  });
});
