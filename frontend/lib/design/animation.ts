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
  // --- F17 (Screen 3/4). Div #53: timings live here, never in config.yaml. ---
  ingestPoll: 1500, // Screen 3: /ingest/status poll interval (ms)
  pipelineStage: 400, // Screen 3: pipeline-stage light-up transition
  forceBar: 600, // Screen 3: SHAP force bar grow-from-zero
  deltaRadar: 600, // Screen 4: delta-radar transition
  compatGauge: 800, // Screen 4: compatibility-projection gauge settle
  // --- F18 (Screen 5). Div #69: timings live here, never in config.yaml. ---
  unknownDrift: 1500, // Screen 5: unknown-driver sphere drift from origin to its position
  spherePulse: 400, // Screen 5: hover sphere pulse
  // NB: the season/camera-focus transition reuses DURATION.styleSpace (1200), already declared above.
  // --- F19 (Screen 6). Div #70: timings live here, never in config.yaml. ---
  replayTick: 900, // Screen 6: base ms per replayed lap at 1× playback (÷ speed for 5× / 10×)
} as const;

// Screen 5: auto-rotating camera speed (master §10.6). Degrees per second.
export const CAMERA_ROTATE_DEG_PER_SEC = 0.2;

export const STAGGER = {
  screen1Axis: 80, // Screen 1: per-axis bloom stagger (bloom effect)
  screen2Axis: 100, // Screen 2: radar per-axis load stagger
} as const;

// Fixed seed for any stochastic D3 layout (beeswarm jitter). Reproducibility (constitution #3):
// a given dataset renders byte-identically every time. No Math.random() in layout.
export const LAYOUT_SEED = 0x9e3779b9;
