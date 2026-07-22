// All motion constants — master §10.1 (F16 spec §4.1/§5). NO timing literal belongs in a component;
// import from here. Values are the master's animation contracts, verbatim.
export const EASING = 'easeCubicInOut' as const; // maps to d3.easeCubicInOut

export const DURATION = {
  chart: 600, // default chart transition
  styleSpace: 1200, // style-space transitions (Screen 5 / F17 — declared here for reuse)
  screen1Bloom: 800, // Screen 1: total radar-bloom duration
  circuitChange: 600, // Screen 1: circuit-selector radar transition
  seasonGhost: 2000, // Screen 2: previous-season ghost hold before it fades
  nlFade: 300, // Screen 2: NL description fade-out/in on season/corner change
} as const;

export const STAGGER = {
  screen1Axis: 80, // Screen 1: per-axis bloom stagger (bloom effect)
  screen2Axis: 100, // Screen 2: radar per-axis load stagger
} as const;

// Fixed seed for any stochastic D3 layout (beeswarm jitter). Reproducibility (constitution #3):
// a given dataset renders byte-identically every time. No Math.random() in layout.
export const LAYOUT_SEED = 0x9e3779b9;
