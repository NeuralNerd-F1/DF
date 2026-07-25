import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { CompareResponse } from '@/lib/api/types';

import { DistinctiveFeatures } from './DistinctiveFeatures';

const compare = (over: Partial<CompareResponse>): CompareResponse =>
  ({
    driver_1_radar: {},
    driver_2_radar: {},
    compatibility_delta: 0,
    ...over,
  }) as unknown as CompareResponse;

describe('DistinctiveFeatures (master §10.5, div #61)', () => {
  it('ranks domains by |delta| and colours by sign', () => {
    render(
      <DistinctiveFeatures compare={compare({ delta_shap: { throttle: 0.5, braking: -0.9, speed: 0.1 } })} />,
    );
    const rows = screen.getAllByTestId('distinctive-row');
    expect(rows[0]).toHaveAttribute('data-domain', 'braking'); // |0.9| first
    expect(rows[0]).toHaveAttribute('data-sign', 'negative');
    expect(rows[1]).toHaveAttribute('data-domain', 'throttle');
  });

  it('renders an EmptyState when there is nothing to distinguish', () => {
    render(<DistinctiveFeatures compare={compare({ delta_shap: {} })} />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });
});
