import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { DistinctiveFeatures } from './DistinctiveFeatures';

describe('DistinctiveFeatures (master §10.5 — per-feature signed-SHAP delta)', () => {
  it('ranks features by |delta|, labels them in plain English, colours by sign', () => {
    render(
      <DistinctiveFeatures
        featureDelta={{ F01: 0.5, F06: -0.9, F17: 0.1 }}
        domainDelta={{ throttle: 1 }}
        driver1="VER"
        driver2="HAM"
      />,
    );
    expect(screen.getByTestId('distinctive-features')).toHaveAttribute('data-mode', 'feature');
    const rows = screen.getAllByTestId('distinctive-row');
    expect(rows[0]).toHaveAttribute('data-key', 'F06'); // |−0.9| first
    expect(rows[0]).toHaveAttribute('data-sign', 'negative');
    expect(rows[0]).toHaveTextContent('peak brake pressure'); // FEATURE_LABELS, not the raw id alone
    expect(rows[1]).toHaveAttribute('data-key', 'F01');
    expect(screen.getByTestId('distinctive-legend')).toHaveTextContent('toward VER');
    expect(screen.queryByTestId('domain-fallback-note')).toBeNull();
  });

  it('caps the list at the top 5', () => {
    const delta = Object.fromEntries(
      Array.from({ length: 9 }, (_, i) => [`F0${i + 1}`, (i + 1) / 10]),
    );
    render(
      <DistinctiveFeatures featureDelta={delta} domainDelta={{}} driver1="A" driver2="B" />,
    );
    expect(screen.getAllByTestId('distinctive-row')).toHaveLength(5);
  });

  it('hover title carries the plain-English meaning with the leading driver', () => {
    render(
      <DistinctiveFeatures
        featureDelta={{ F27: -0.4 }}
        domainDelta={{}}
        driver1="VER"
        driver2="HAM"
      />,
    );
    expect(screen.getByTestId('distinctive-row')).toHaveAttribute(
      'title',
      expect.stringContaining('HAM'),
    );
    expect(screen.getByTestId('distinctive-row')).toHaveAttribute(
      'title',
      expect.stringContaining('tyre-age throttle drift'),
    );
  });

  it('falls back to the served per-domain delta, labelled honestly, when signed SHAP is absent', () => {
    render(
      <DistinctiveFeatures
        featureDelta={null}
        domainDelta={{ throttle: 0.5, braking: -0.9 }}
        driver1="VER"
        driver2="HAM"
      />,
    );
    expect(screen.getByTestId('distinctive-features')).toHaveAttribute('data-mode', 'domain');
    expect(screen.getByTestId('domain-fallback-note')).toBeInTheDocument();
    const rows = screen.getAllByTestId('distinctive-row');
    expect(rows[0]).toHaveAttribute('data-key', 'braking');
  });

  it('renders an EmptyState when there is nothing to distinguish', () => {
    render(<DistinctiveFeatures featureDelta={null} domainDelta={{}} driver1="A" driver2="B" />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });
});
