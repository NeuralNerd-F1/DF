import { create } from 'zustand';

// Client-side playback state for Screen 6 (master §10.7, div #71). The whole replay is pre-fetched;
// this store holds ONLY the playback position — Play/Pause/Speed/Scrub operate over the cached laps.
// The advancing interval lives in lib/replay/usePlayback.ts, not here (keeps the store side-effect-free).
export type ReplaySpeed = 1 | 5 | 10;

type ReplayState = {
  playing: boolean;
  speed: ReplaySpeed;
  lapIndex: number;
  setPlaying: (playing: boolean) => void;
  setSpeed: (speed: ReplaySpeed) => void;
  setLapIndex: (lapIndex: number) => void;
  reset: () => void;
};

export const useReplay = create<ReplayState>((set) => ({
  playing: false,
  speed: 1,
  lapIndex: 0,
  setPlaying: (playing) => set({ playing }),
  setSpeed: (speed) => set({ speed }),
  setLapIndex: (lapIndex) => set({ lapIndex }),
  reset: () => set({ playing: false, speed: 1, lapIndex: 0 }),
}));
