/**
 * Integrators, in double precision, matching OpenMM step for step.
 *
 * Two of them, and the engine needs both. `LangevinMiddleIntegrator` --
 * which `LangevinIntegrator` has been since OpenMM 8.2, and so what the
 * notebook runs -- is the production one. `VerletIntegrator` is what the NVE
 * drift gate runs, because OpenMM reports the half-step-shifted kinetic energy
 * for it and that is the conserved quantity. See docs/DYNAMICS.md.
 *
 * Both are leapfrog: one full-step kick, then a drift. They differ in only two
 * places, and neither is where one would guess:
 *
 *   * LFMiddle RATTLE-projects the velocities immediately after the kick, and
 *     splits its drift around a thermostat step. Verlet does neither.
 *   * Their reported kinetic energies use different conventions, which is a
 *     property of the integrator rather than a choice (src/reference/kinetic.ts).
 *
 * Transcribed from `ReferenceVerletDynamics::update` and
 * `ReferenceLangevinMiddleDynamics::update`.
 */

import { BOLTZMANN_KJ_PER_MOL_K } from "../engine/units.js";
import type { OpenMMSystem } from "../engine/system.js";
import {
  constrainPositions,
  constrainVelocities,
  inverseMasses,
  type ConstraintResult,
} from "./constraints.js";
import {
  splitConstraints,
  type ConstraintSplit,
} from "../engine/constraint-clusters.js";

export interface IntegratorState {
  /** nm, xyz interleaved. Modified in place. */
  positions: Float64Array;
  /** nm/ps, xyz interleaved. Modified in place. */
  velocities: Float64Array;
}

export interface StepResult {
  readonly positionConstraints: ConstraintResult;
  readonly velocityConstraints: ConstraintResult | null;
}

/** A source of standard normal deviates, so a run can be made reproducible. */
export type Normals = () => number;

/**
 * One velocity-Verlet (leapfrog) step.
 *
 *   v += dt F/m ;  x' = x + dt v ;  SHAKE(x, x') ;  v = (x' - x)/dt ;  x = x'
 *
 * The velocity comes from the *constrained* displacement, which is what folds
 * the constraint's effect back into it.
 */
export function verletStep(
  system: OpenMMSystem,
  state: IntegratorState,
  forces: Float64Array,
  timeStep: number,
  constraintTolerance: number,
  split: ConstraintSplit = splitConstraints(system),
): StepResult {
  const inverse = inverseMasses(system);
  const previous = Float64Array.from(state.positions);
  const trial = new Float64Array(state.positions.length);
  for (let i = 0; i < system.particleCount; i++) {
    const inverseMass = inverse[i] as number;
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      if (inverseMass === 0) {
        trial[at] = previous[at] as number;
        continue;
      }
      const velocity =
        (state.velocities[at] as number) +
        inverseMass * (forces[at] as number) * timeStep;
      state.velocities[at] = velocity;
      trial[at] = (previous[at] as number) + velocity * timeStep;
    }
  }
  const positionConstraints = constrainPositions(
    system,
    previous,
    trial,
    constraintTolerance,
    split,
  );
  const scale = 1 / timeStep;
  for (let i = 0; i < system.particleCount; i++) {
    if ((inverse[i] as number) === 0) continue;
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      state.velocities[at] =
        scale * ((trial[at] as number) - (previous[at] as number));
      state.positions[at] = trial[at] as number;
    }
  }
  return { positionConstraints, velocityConstraints: null };
}

export interface LangevinOptions {
  readonly timeStep: number;
  /** K */
  readonly temperature: number;
  /** ps^-1 */
  readonly friction: number;
  readonly constraintTolerance: number;
  /**
   * Standard normal deviates, three per moving particle per step.
   *
   * With `friction = 0` the noise scale is zero and this is never called, which
   * is what makes a zero-friction run deterministic and usable as an oracle.
   */
  readonly normals?: Normals;
  /** Precomputed constraint classification, to avoid redoing it every step. */
  readonly split?: ConstraintSplit;
}

/**
 * One LFMiddle step: kick, RATTLE, half drift, thermostat, half drift, SHAKE.
 *
 * `vscale = exp(-dt*friction)` and `noisescale = sqrt(1 - vscale^2)`, so a
 * friction of zero leaves the velocities untouched in the middle and the step
 * reduces to the same leapfrog Verlet performs -- bar the velocity projection
 * after the kick, which Verlet does not do.
 */
export function langevinMiddleStep(
  system: OpenMMSystem,
  state: IntegratorState,
  forces: Float64Array,
  options: LangevinOptions,
): StepResult {
  const { timeStep, temperature, friction, constraintTolerance } = options;
  const split = options.split ?? splitConstraints(system);
  const inverse = inverseMasses(system);
  const halfStep = 0.5 * timeStep;
  const kT = BOLTZMANN_KJ_PER_MOL_K * temperature;
  const velocityScale = Math.exp(-timeStep * friction);
  const noiseScale = Math.sqrt(1 - velocityScale * velocityScale);
  const normals = options.normals;
  if (noiseScale > 0 && normals === undefined) {
    throw new Error(
      "langevinMiddleStep needs a source of normal deviates when friction is " +
        "non-zero; pass `normals`. A run must record its seed, so the engine " +
        "never reaches for an unseeded generator.",
    );
  }

  // Kick, then project the velocities.
  for (let i = 0; i < system.particleCount; i++) {
    const inverseMass = inverse[i] as number;
    if (inverseMass === 0) continue;
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      state.velocities[at] =
        (state.velocities[at] as number) +
        timeStep * inverseMass * (forces[at] as number);
    }
  }
  const velocityConstraints = constrainVelocities(
    system,
    state.positions,
    state.velocities,
    constraintTolerance,
    split,
  );

  // Half drift, thermostat, half drift. `unconstrained` is OpenMM's `oldx`:
  // the trial positions before SHAKE, which is what the velocity correction is
  // measured against.
  const trial = Float64Array.from(state.positions);
  for (let i = 0; i < system.particleCount; i++) {
    const inverseMass = inverse[i] as number;
    if (inverseMass === 0) continue;
    const noise = noiseScale * Math.sqrt(kT * inverseMass);
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      trial[at] = (trial[at] as number) + (state.velocities[at] as number) * halfStep;
      state.velocities[at] =
        velocityScale * (state.velocities[at] as number) +
        (noise === 0 ? 0 : noise * (normals as Normals)());
      trial[at] = (trial[at] as number) + (state.velocities[at] as number) * halfStep;
    }
  }
  const unconstrained = Float64Array.from(trial);
  const positionConstraints = constrainPositions(
    system,
    state.positions,
    trial,
    constraintTolerance,
    split,
  );

  // What the projection moved, over dt, is the velocity it implies.
  const scale = 1 / timeStep;
  for (let i = 0; i < system.particleCount; i++) {
    if ((inverse[i] as number) === 0) continue;
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      state.velocities[at] =
        (state.velocities[at] as number) +
        scale * ((trial[at] as number) - (unconstrained[at] as number));
      state.positions[at] = trial[at] as number;
    }
  }
  return { positionConstraints, velocityConstraints };
}
