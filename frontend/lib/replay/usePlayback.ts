'use client';

import { useEffect } from 'react';

import { DURATION } from '@/lib/design/animation';
import { useReplay } from '@/lib/state/replay';

// The playback engine (master §10.7, div #71). While `playing`, advances lapIndex one step every
// DURATION.replayTick / speed ms over the pre-fetched laps. It STOPS (and the effect cleanup clears
// the interval) at the terminal lap, on pause, on speed change (re-armed), and on unmount — the
// cleanup is the no-leak / no-runaway guarantee (§7 criterion 8). Interval id is local to the effect.
export function usePlayback(lapCount: number): void {
  const playing = useReplay((s) => s.playing);
  const speed = useReplay((s) => s.speed);
  const setPlaying = useReplay((s) => s.setPlaying);
  const setLapIndex = useReplay((s) => s.setLapIndex);

  useEffect(() => {
    if (!playing || lapCount <= 0) return;
    const period = Math.max(1, Math.round(DURATION.replayTick / speed));
    const id = setInterval(() => {
      // Read the latest index at tick time from the store (avoids stale-closure capture of lapIndex).
      const { lapIndex } = useReplay.getState();
      if (lapIndex >= lapCount - 1) {
        setPlaying(false); // terminal lap → stop; the effect re-runs and cleanup clears this timer
        return;
      }
      setLapIndex(lapIndex + 1);
    }, period);
    return () => clearInterval(id);
  }, [playing, speed, lapCount, setPlaying, setLapIndex]);
}
