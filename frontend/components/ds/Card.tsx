import type { ReactNode } from 'react';

// Base card surface (master §10.1 card token). `dashed` is the LOW_CONFIDENCE treatment (§10.2).
export function Card({
  children,
  className = '',
  dashed = false,
}: {
  children: ReactNode;
  className?: string;
  dashed?: boolean;
}) {
  const borderStyle = dashed ? 'border-dashed border-border' : 'border-border';
  return (
    <div className={`rounded-lg border bg-card ${borderStyle} ${className}`}>{children}</div>
  );
}
