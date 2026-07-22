'use client';

import type { ButtonHTMLAttributes } from 'react';

// Small control button (sort/toggle). `active` marks the selected state (master §10.1 active accent).
export function Button({
  active = false,
  className = '',
  type = 'button',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  const tone = active
    ? 'border-accent text-accent'
    : 'border-border text-text-secondary hover:border-accent hover:text-text-primary';
  return (
    <button
      type={type}
      className={`rounded border px-3 py-1 font-mono text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${tone} ${className}`}
      {...props}
    />
  );
}
