import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ConfidenceBadge } from './ConfidenceBadge';
import { ConfidenceBar } from './ConfidenceBar';
import { EmptyState } from './EmptyState';
import { ErrorCard } from './ErrorCard';
import { Skeleton } from './Skeleton';

describe('ErrorCard (master §10.1 error state)', () => {
  it('shows the error_code in monospace, a red left border, and Retry for a retryable code', () => {
    const onRetry = vi.fn();
    render(<ErrorCard code="DB_UNAVAILABLE" onRetry={onRetry} />);
    expect(screen.getByTestId('error-code')).toHaveTextContent('DB_UNAVAILABLE');
    expect(screen.getByTestId('error-code')).toHaveClass('font-mono');
    expect(screen.getByTestId('error-card')).toHaveClass('border-l-accent');
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('offers no Retry for a non-retryable 404 NOT_FOUND', () => {
    render(<ErrorCard code="NOT_FOUND" onRetry={() => undefined} />);
    expect(screen.queryByRole('button', { name: /retry/i })).toBeNull();
  });
});

describe('ConfidenceBar (div #46 — colour from state, not a numeric re-threshold)', () => {
  it('colours off confidence_state even when the number would disagree', () => {
    const { rerender } = render(<ConfidenceBar confidence={0.99} state="MEDIUM" />);
    // 0.99 numeric would read "green" if re-thresholded — but MEDIUM is authoritative -> amber.
    expect(screen.getByTestId('confidence-bar')).toHaveAttribute('data-tone', 'medium');
    rerender(<ConfidenceBar confidence={0.1} state="HIGH" />);
    expect(screen.getByTestId('confidence-bar')).toHaveAttribute('data-tone', 'high');
    rerender(<ConfidenceBar confidence={0.7} state="LOW_CONFIDENCE" />);
    expect(screen.getByTestId('confidence-bar')).toHaveAttribute('data-tone', 'low');
  });

  it('sizes the fill from the numeric confidence', () => {
    render(<ConfidenceBar confidence={0.5} state="MEDIUM" />);
    expect(screen.getByTestId('confidence-fill')).toHaveAttribute('width', '50');
  });
});

describe('ConfidenceBadge', () => {
  it('renders nothing for a confident state', () => {
    const { container } = render(<ConfidenceBadge state="HIGH" />);
    expect(container.firstChild).toBeNull();
  });

  it('warns for an uncertainty state', () => {
    render(<ConfidenceBadge state="LOW_CONFIDENCE" />);
    expect(screen.getByTestId('confidence-badge')).toHaveTextContent('LOW CONFIDENCE');
  });
});

describe('Skeleton + EmptyState', () => {
  it('skeleton is a status region (not a spinner) and empty is distinct from error', () => {
    const { unmount } = render(<Skeleton />);
    expect(screen.getByTestId('skeleton')).toHaveAttribute('role', 'status');
    unmount();
    render(<EmptyState message="No driver data yet." />);
    expect(screen.getByTestId('empty-state')).toHaveTextContent('No driver data yet.');
  });
});
