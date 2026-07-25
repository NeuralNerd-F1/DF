import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { PipelineProgress } from './PipelineProgress';

describe('PipelineProgress (master §10.4, div #55)', () => {
  it('marks every stage done on COMPLETE', () => {
    render(<PipelineProgress status="COMPLETE" progress={1} />);
    const stages = screen.getAllByTestId('pipeline-stage');
    expect(stages).toHaveLength(5);
    expect(stages.every((s) => s.getAttribute('data-state') === 'done')).toBe(true);
  });

  it('turns the reached stage red with the error_detail on FAILED (no fabricated timings)', () => {
    render(<PipelineProgress status="FAILED" progress={0.5} errorDetail="INGEST_TASK_FAILED" />);
    const failed = screen.getAllByTestId('pipeline-stage').find((s) => s.getAttribute('data-state') === 'failed');
    expect(failed).toBeDefined();
    expect(screen.getByTestId('pipeline-error')).toHaveTextContent('INGEST_TASK_FAILED');
  });

  it('shows all stages pending when idle', () => {
    render(<PipelineProgress status="IDLE" progress={0} />);
    const stages = screen.getAllByTestId('pipeline-stage');
    expect(stages.every((s) => s.getAttribute('data-state') === 'pending')).toBe(true);
  });
});
