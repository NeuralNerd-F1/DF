import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useReplay } from '@/lib/state/replay';

import { usePlayback } from './usePlayback';

beforeEach(() => {
  useReplay.getState().reset();
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
  useReplay.getState().reset();
});

describe('usePlayback (div #71)', () => {
  it('advances lapIndex one step per tick while playing', () => {
    renderHook(() => usePlayback(3));
    act(() => useReplay.getState().setPlaying(true));
    expect(useReplay.getState().lapIndex).toBe(0);
    act(() => vi.advanceTimersByTime(900)); // DURATION.replayTick at 1×
    expect(useReplay.getState().lapIndex).toBe(1);
    act(() => vi.advanceTimersByTime(900));
    expect(useReplay.getState().lapIndex).toBe(2);
  });

  it('ticks faster at higher speed (replayTick / speed)', () => {
    renderHook(() => usePlayback(20));
    act(() => {
      useReplay.getState().setSpeed(10);
      useReplay.getState().setPlaying(true);
    });
    act(() => vi.advanceTimersByTime(90)); // 900 / 10
    expect(useReplay.getState().lapIndex).toBe(1);
  });

  it('stops (playing→false) at the terminal lap', () => {
    renderHook(() => usePlayback(2));
    act(() => useReplay.getState().setPlaying(true));
    act(() => vi.advanceTimersByTime(900)); // 0 → 1 (terminal)
    expect(useReplay.getState().lapIndex).toBe(1);
    act(() => vi.advanceTimersByTime(900)); // tick sees terminal → stop
    expect(useReplay.getState().playing).toBe(false);
    expect(useReplay.getState().lapIndex).toBe(1);
  });

  it('stops advancing once paused (interval cleared)', () => {
    renderHook(() => usePlayback(5));
    act(() => useReplay.getState().setPlaying(true));
    act(() => vi.advanceTimersByTime(900)); // → 1
    act(() => useReplay.getState().setPlaying(false));
    const before = useReplay.getState().lapIndex;
    act(() => vi.advanceTimersByTime(3000));
    expect(useReplay.getState().lapIndex).toBe(before);
  });

  it('clears the timer on unmount — no leak/runaway (criterion 8)', () => {
    const { unmount } = renderHook(() => usePlayback(10));
    act(() => useReplay.getState().setPlaying(true));
    act(() => vi.advanceTimersByTime(900)); // → 1
    unmount();
    const before = useReplay.getState().lapIndex;
    act(() => vi.advanceTimersByTime(9000)); // nothing should advance after unmount
    expect(useReplay.getState().lapIndex).toBe(before);
  });
});
