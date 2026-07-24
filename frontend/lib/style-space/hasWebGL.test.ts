import { describe, expect, it } from 'vitest';

import { hasWebGL } from './hasWebGL';

describe('hasWebGL (master §10.6 fallback boundary, div #68)', () => {
  it('returns false in jsdom (no WebGL context) → selects the D3 2D fallback', () => {
    // jsdom has no WebGL implementation, so getContext('webgl2'|'webgl') is null.
    expect(hasWebGL()).toBe(false);
  });
});
