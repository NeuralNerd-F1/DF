import { afterEach, describe, expect, it } from 'vitest';

import { useReplay } from './replay';

afterEach(() => useReplay.getState().reset());

describe('useReplay store', () => {
  it('defaults to paused, 1× speed, lap 0', () => {
    const s = useReplay.getState();
    expect(s.playing).toBe(false);
    expect(s.speed).toBe(1);
    expect(s.lapIndex).toBe(0);
  });

  it('sets playing / speed / lapIndex', () => {
    useReplay.getState().setPlaying(true);
    useReplay.getState().setSpeed(10);
    useReplay.getState().setLapIndex(7);
    const s = useReplay.getState();
    expect(s.playing).toBe(true);
    expect(s.speed).toBe(10);
    expect(s.lapIndex).toBe(7);
  });

  it('reset() rewinds everything to the defaults', () => {
    useReplay.getState().setPlaying(true);
    useReplay.getState().setSpeed(5);
    useReplay.getState().setLapIndex(4);
    useReplay.getState().reset();
    const s = useReplay.getState();
    expect(s).toMatchObject({ playing: false, speed: 1, lapIndex: 0 });
  });
});
