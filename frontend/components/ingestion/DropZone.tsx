'use client';

import { useRef, useState } from 'react';

// Drag-and-drop CSV zone (master §10.4 Manual Upload tab). The dashed border pulses while a file is
// being dragged over it. Clicking opens the file picker. The parent owns what happens on select —
// currently POST /ingest/upload (a 503 stub → F05, div #58), surfaced honestly by the parent.
export function DropZone({ onFile, disabled = false }: { onFile: (f: File) => void; disabled?: boolean }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div
      data-testid="dropzone"
      data-dragging={dragging}
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (disabled) return;
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
      onClick={() => {
        if (!disabled) inputRef.current?.click();
      }}
      onKeyDown={(e) => {
        if (!disabled && (e.key === 'Enter' || e.key === ' ')) inputRef.current?.click();
      }}
      className={`cursor-pointer rounded-lg border-2 border-dashed p-8 text-center text-sm text-text-secondary transition-colors ${
        dragging ? 'animate-pulse border-accent' : 'border-border hover:border-accent'
      } ${disabled ? 'cursor-not-allowed opacity-40' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        data-testid="file-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      Drag &amp; drop a telemetry CSV, or click to choose a file.
    </div>
  );
}
