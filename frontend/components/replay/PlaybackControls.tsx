'use client';

import { Button } from '@/components/ds/Button';
import type { ReplaySpeed } from '@/lib/state/replay';
import { useReplay } from '@/lib/state/replay';

// Play / Pause / Speed / Scrub (master §10.7, div #71). All store-driven; the advancing interval
// lives in usePlayback. Scrub sets lapIndex directly. Restarting from a paused terminal lap resets
// the position first so Play always makes progress.
const SPEEDS: ReplaySpeed[] = [1, 5, 10];

export function PlaybackControls({
  lapCount,
  currentLapNumber,
}: {
  lapCount: number;
  currentLapNumber: number | null;
}) {
  const playing = useReplay((s) => s.playing);
  const speed = useReplay((s) => s.speed);
  const lapIndex = useReplay((s) => s.lapIndex);
  const setPlaying = useReplay((s) => s.setPlaying);
  const setSpeed = useReplay((s) => s.setSpeed);
  const setLapIndex = useReplay((s) => s.setLapIndex);

  const atEnd = lapIndex >= lapCount - 1;

  return (
    <div className="flex flex-wrap items-center gap-3" data-testid="playback-controls">
      <Button
        active={playing}
        onClick={() => {
          if (!playing && atEnd) setLapIndex(0); // restart from the top if paused at the terminal lap
          setPlaying(!playing);
        }}
        data-testid="playback-toggle"
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? 'Pause' : 'Play'}
      </Button>

      <div className="flex items-center gap-1" role="group" aria-label="Speed">
        {SPEEDS.map((s) => (
          <Button key={s} active={speed === s} onClick={() => setSpeed(s)} data-testid={`speed-${s}`}>
            {s}×
          </Button>
        ))}
      </div>

      <input
        type="range"
        min={0}
        max={Math.max(0, lapCount - 1)}
        value={lapIndex}
        onChange={(e) => setLapIndex(Number(e.target.value))}
        className="flex-1 accent-accent"
        aria-label="Scrub"
        data-testid="scrub"
      />
      <span className="font-mono text-xs text-text-secondary" data-testid="lap-readout">
        {currentLapNumber != null ? `Lap ${currentLapNumber}` : '—'} ({lapIndex + 1}/{lapCount})
      </span>
    </div>
  );
}
