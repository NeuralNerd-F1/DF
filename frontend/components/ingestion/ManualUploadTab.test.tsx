import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

vi.mock('@/lib/api/endpoints', () => ({ postUpload: vi.fn() }));

import { postUpload } from '@/lib/api/endpoints';

import { ManualUploadTab } from './ManualUploadTab';

const err = (code: string, status: number, detail = '') => ({ ok: false, code, detail, status });

afterEach(() => vi.clearAllMocks());

describe('ManualUploadTab (master §10.4, div #58)', () => {
  it('renders an F05 unavailable notice before any upload', () => {
    render(<ManualUploadTab />);
    expect(screen.getByTestId('empty-state')).toHaveTextContent(/arrives in F05/i);
  });

  it('POSTs the real endpoint and surfaces the honest 503 — never a fabricated cleaning report', async () => {
    (postUpload as Mock).mockResolvedValue(err('HEALTH_FAIL', 503, 'upload ships in F05'));
    render(<ManualUploadTab />);
    const file = new File(['a,b\n1,2'], 'laps.csv', { type: 'text/csv' });
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [file] } });
    await waitFor(() => expect(screen.getByTestId('upload-unavailable')).toBeInTheDocument());
    expect(screen.getByTestId('error-code')).toHaveTextContent('HEALTH_FAIL');
    expect(screen.queryByTestId('cleaning-report')).toBeNull();
    expect(postUpload as Mock).toHaveBeenCalledOnce();
  });
});
