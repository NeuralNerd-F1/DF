import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { useReplay } from '@/lib/state/replay';

import { PlaybackControls } from './PlaybackControls';

beforeEach(() => useReplay.getState().reset());
afterEach(() => useReplay.getState().reset());

describe('PlaybackControls (div #71 — store-driven)', () => {
  it('Play toggles the store playing flag and the button label', () => {
    render(<PlaybackControls lapCount={5} currentLapNumber={1} />);
    fireEvent.click(screen.getByTestId('playback-toggle'));
    expect(useReplay.getState().playing).toBe(true);
    expect(screen.getByTestId('playback-toggle')).toHaveTextContent('Pause');
    fireEvent.click(screen.getByTestId('playback-toggle'));
    expect(useReplay.getState().playing).toBe(false);
  });

  it('Speed buttons set the store speed (1/5/10)', () => {
    render(<PlaybackControls lapCount={5} currentLapNumber={1} />);
    fireEvent.click(screen.getByTestId('speed-10'));
    expect(useReplay.getState().speed).toBe(10);
    fireEvent.click(screen.getByTestId('speed-5'));
    expect(useReplay.getState().speed).toBe(5);
  });

  it('Scrub sets lapIndex directly', () => {
    render(<PlaybackControls lapCount={5} currentLapNumber={1} />);
    fireEvent.change(screen.getByTestId('scrub'), { target: { value: '3' } });
    expect(useReplay.getState().lapIndex).toBe(3);
  });

  it('restarts from the top when Play is pressed at the terminal lap', () => {
    useReplay.getState().setLapIndex(4); // terminal for lapCount 5
    render(<PlaybackControls lapCount={5} currentLapNumber={5} />);
    fireEvent.click(screen.getByTestId('playback-toggle'));
    expect(useReplay.getState().lapIndex).toBe(0);
    expect(useReplay.getState().playing).toBe(true);
  });
});
