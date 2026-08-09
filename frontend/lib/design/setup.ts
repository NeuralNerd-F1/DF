// Setup-parameter display metadata (system input #2 — "estimate setup tendencies"). Pure and
// jsdom-free so the phrasing rules are unit-testable.
//
// HONESTY RULES (spec F11 §1: "a starting direction, not an engineering specification"):
// - A tendency states the served value's position vs the SERVED grid reference (`grid_mean` — the
//   model's own mean recommendation across the known grid, same circuit context). It is never a
//   claim about a real team's setup, and it renders only when both numbers exist.
// - The more/less phrases describe what the PARAMETER does physically (an engineering fact about
//   the control), never why the model chose it — the regressor's inputs are opaque UMAP dims.
// - CATEGORY parameters get no tendency (their served grid_mean is null by design), WIDE
//   parameters get none (no point value to compare), and equal formatted values read as
//   "in line with the grid average", not a manufactured direction.
import type { SetupParameterSchema } from '@/lib/api/types';

export type SetupMeta = {
  label: string;
  more: string; // reads after "…than the grid average — " when value > grid_mean
  less: string; // when value < grid_mean
};

export const SETUP_META: Readonly<Record<string, SetupMeta>> = {
  front_wing_angle: {
    label: 'Front wing angle',
    more: 'more front downforce — sharper turn-in',
    less: 'less front downforce — calmer front end',
  },
  rear_wing_angle: {
    label: 'Rear wing angle',
    more: 'more rear downforce — stability over top speed',
    less: 'less rear downforce — top speed over stability',
  },
  brake_bias: {
    label: 'Brake bias',
    more: 'more front-biased braking',
    less: 'more rearward brake balance',
  },
  differential_ramp_entry: {
    label: 'Differential ramp (entry)',
    more: 'more locked diff into corners',
    less: 'freer diff into corners',
  },
  differential_ramp_exit: {
    label: 'Differential ramp (exit)',
    more: 'more locked diff out of corners',
    less: 'freer diff out of corners',
  },
  throttle_map_curve: {
    label: 'Throttle map curve',
    more: 'more aggressive throttle response',
    less: 'more progressive throttle response',
  },
  suspension_stiffness_front: {
    label: 'Front suspension',
    more: 'stiffer front suspension',
    less: 'softer front suspension',
  },
  suspension_stiffness_rear: {
    label: 'Rear suspension',
    more: 'stiffer rear suspension',
    less: 'softer rear suspension',
  },
  tyre_warmup_laps: {
    label: 'Tyre warm-up laps',
    more: 'longer tyre preparation',
    less: 'shorter tyre preparation',
  },
  brake_duct_aperture: {
    label: 'Brake duct aperture',
    more: 'more brake cooling',
    less: 'less brake cooling — lower drag',
  },
};

// The tendency sentence for one parameter, or null when no honest direction exists. The direction
// is decided on the FORMATTED values (1 dp / integer per kind) so a sub-display-precision delta
// never manufactures a direction the user cannot see in the numbers.
export function tendencyText(p: SetupParameterSchema): string | null {
  const meta = SETUP_META[p.name];
  if (!meta) return null;
  if (typeof p.value !== 'number' || typeof p.grid_mean !== 'number') return null;
  const digits = String(p.kind) === 'INDEX' || String(p.kind) === 'INTEGER_LAPS' ? 0 : 1;
  const v = p.value.toFixed(digits);
  const g = p.grid_mean.toFixed(digits);
  if (v === g) return `in line with the grid average (${g})`;
  const dir = p.value > p.grid_mean ? meta.more : meta.less;
  return `${dir} (vs grid average ${g})`;
}
