import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

// Minimal `three` mock: WebGLRenderer throws (no WebGL), exercising the scene's try/catch → the
// silent-fallback path (master §10.6 / §6 failure mode). Only the classes reached before the throw
// need stubbing (Scene, PerspectiveCamera).
vi.mock('three', () => ({
  Scene: class {},
  PerspectiveCamera: class {
    position = { set: () => {} };
    updateProjectionMatrix() {}
  },
  WebGLRenderer: class {
    constructor() {
      throw new Error('WebGL unavailable');
    }
  },
}));
vi.mock('three/examples/jsm/controls/OrbitControls.js', () => ({
  OrbitControls: class {
    update() {}
    dispose() {}
  },
}));

import { StyleSpaceScene } from './StyleSpaceScene';

const nodes = [{ code: 'VER', coords2d: [0, 0] as [number, number], coords5d: [1, 0, 0], team: 'Red Bull' }];

describe('StyleSpaceScene (master §10.6, div #68)', () => {
  it('renders its mount div and calls onInitError when WebGL init throws (silent fallback)', () => {
    const onInitError = vi.fn();
    render(
      <StyleSpaceScene
        nodes={nodes}
        teammates={[]}
        showTeammates={false}
        selected={null}
        onSelect={() => {}}
        onHover={() => {}}
        onInitError={onInitError}
      />,
    );
    // The component never throws into React — the container mounts...
    expect(screen.getByTestId('style-space-scene')).toBeInTheDocument();
    // ...and the WebGL failure degrades via the callback (parent then switches to the D3 2D view).
    expect(onInitError).toHaveBeenCalled();
  });
});
