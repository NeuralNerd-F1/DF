// All motion constants — master §10.1 (F16 spec §4.1/§5). NO timing literal belongs in a component;
// import from here. Values are the master's animation contracts, verbatim.
export const DURATION = {
  chart: 600, // default chart transition
  styleSpace: 1200, // style-space transitions (the Driver Profile's style map)
  screen1Bloom: 800, // Screen 1: total radar-bloom duration
  deltaRadar: 600, // Screen 4: delta-radar transition
  replayTick: 900, // Replay: base ms per replayed lap at 1× playback (÷ speed for 5× / 10×)
} as const;

export const STAGGER = {
  screen1Axis: 80, // Screen 1: per-axis bloom stagger (bloom effect)
  screen2Axis: 100, // Screen 2: radar per-axis load stagger
} as const;
