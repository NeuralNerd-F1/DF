// WebGL capability probe (master §10.6). Returns false when WebGL is unavailable — including in
// jsdom (which has no WebGL context) and during SSR — which selects the silent D3 2D fallback.
// Never throws.
export function hasWebGL(): boolean {
  if (typeof document === 'undefined') return false; // SSR — render the 2D path
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    return gl != null;
  } catch {
    return false;
  }
}
