import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const refresh = vi.fn();
vi.mock('next/navigation', () => ({ useRouter: () => ({ refresh }) }));

import { PageError } from './PageError';

describe('PageError (HIGH-1 — the page-level Retry is actually wired)', () => {
  it('wires Retry to router.refresh() for a retryable code', () => {
    render(<PageError code="DB_UNAVAILABLE" detail="database down" />);
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(refresh).toHaveBeenCalledOnce();
  });
});
