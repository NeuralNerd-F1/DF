'use client';

import { Button } from '@/components/ds/Button';

// Teammate proximity-lines toggle (master §10.6).
export function TeammateLinesToggle({
  value,
  onChange,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <Button
      active={value}
      aria-pressed={value}
      onClick={() => onChange(!value)}
      data-testid="teammate-toggle"
    >
      Teammate lines
    </Button>
  );
}
