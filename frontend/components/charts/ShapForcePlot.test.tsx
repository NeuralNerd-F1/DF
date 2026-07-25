import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ShapForcePlot } from './ShapForcePlot';

describe('ShapForcePlot (master §10.4, div #56)', () => {
  it('renders one signed bar per feature: red toward, blue away', () => {
    render(
      <ShapForcePlot
        available
        shapValues={{ F01: 0.4, F02: -0.2 }}
        featureValues={{ F01: 12, F02: 3 }}
        animate={false}
      />,
    );
    const rows = screen.getAllByTestId('force-row');
    expect(rows).toHaveLength(2);
    expect(screen.getByText('F01').closest('[data-testid="force-row"]')).toHaveAttribute(
      'data-sign',
      'toward',
    );
    expect(screen.getByText('F02').closest('[data-testid="force-row"]')).toHaveAttribute(
      'data-sign',
      'away',
    );
  });

  it('sorts by |shap| descending and caps at maxBars', () => {
    render(
      <ShapForcePlot
        available
        shapValues={{ A: 0.1, B: -0.9, C: 0.5 }}
        maxBars={2}
        animate={false}
      />,
    );
    const rows = screen.getAllByTestId('force-row');
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveAttribute('data-feature', 'B'); // |0.9| first
    expect(rows[1]).toHaveAttribute('data-feature', 'C'); // |0.5| second
  });

  it('shows the request feature value in the bar hover title', () => {
    const { container } = render(
      <ShapForcePlot available shapValues={{ F01: 0.3 }} featureValues={{ F01: 12 }} animate={false} />,
    );
    const title = container.querySelector('title');
    expect(title?.textContent).toContain('F01: value=12, SHAP=0.300 (toward)');
  });

  it('renders an EmptyState (not a blank plot) when SHAP is unavailable', () => {
    render(<ShapForcePlot available={false} shapValues={null} animate={false} />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.queryByTestId('shap-force')).toBeNull();
  });

  it('renders an EmptyState when shap_values is empty even if available', () => {
    render(<ShapForcePlot available shapValues={{}} animate={false} />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });
});
