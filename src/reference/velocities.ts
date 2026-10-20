/**
 * Initial velocities, as `Context.setVelocitiesToTemperature` sets them.
 *
 * Transcribed from `Context::setVelocitiesToTemperature` and
 * `Integrator::getVelocitiesForTemperature`, both read rather than recalled,
 * because the obvious implementation is missing two steps.
 *
 * 🔴 **The draw is shifted by `(dt/2) F/m`, for both integrators.**
 * `setVelocitiesToTemperature` draws from Maxwell-Boltzmann, then evaluates the
 * forces and subtracts `(offset/m) F`, where `offset` is the integrator's
 * velocity time offset. It is `dt/2` for `VerletIntegrator` as much as for
 * `LangevinIntegrator`: OpenMM's Verlet is leapfrog internally and stores its
 * velocities at `t - dt/2`, which is the same fact as its *reporting* a
 * half-step-shifted kinetic energy (docs/DYNAMICS.md). Omit the shift and the
 * run starts half a step out of phase with the scheme that will propagate it.
 *
 * The evidence is easy to read backwards, and I did. The two integrators
 * produce *identical* statistics -- 306.83 K and 0.7090 variance per kT on
 * 1L2Y, to every digit -- which looks like neither of them shifting. They are
 * identical because both shift by the same amount. What settles it is the
 * variance of the draw with the constraints stripped out, where
 *
 *     mean(m v^2)/kT  =  1 + (dt/2)^2 mean(F^2/m)/kT
 *
 * and 3HTB measures **1.4189** against a prediction of **1.4173**, with the
 * velocities correlated +0.52 with `-(dt/2) F/m`. Unshifted it would be 1.0 and
 * uncorrelated. `tools/capture_velocities.py` records both, so the next reader
 * does not have to take either reading on trust.
 *
 * `timeOffset` has no default for that reason: a caller passes the integrator's
 * offset, which is `dt/2` for everything this engine runs, and passing zero is
 * a decision rather than an omission.
 *
 * 🔴 **The constraint projection uses 1e-5, not the integrator's tolerance.**
 * That is hard-coded in `setVelocitiesToTemperature` and has nothing to do with
 * `setConstraintTolerance(1e-6)`. Using the integrator's would be a defensible
 * choice and a different starting state.
 *
 * The deviates cannot match OpenMM's: it draws from SFMT by the Marsaglia polar
 * method, consuming them only for *moving* particles, and that stream is not
 * reproducible from outside. Ours come from the counter-based generator on its
 * own domain, indexed by particle and axis, and the seed is recorded. The
 * distribution is the same; the sample is not. See deviation 15.
 */

import { BOLTZMANN_KJ_PER_MOL_K } from "../engine/units.js";
import type { OpenMMSystem } from "../engine/system.js";
import {
  splitConstraints,
  type ConstraintSplit,
} from "../engine/constraint-clusters.js";
import { constrainVelocities } from "./constraints.js";
import { domainNormal, VELOCITY_DOMAIN } from "./random.js";

/** The tolerance `setVelocitiesToTemperature` projects with, which is its own. */
export const VELOCITY_CONSTRAINT_TOLERANCE = 1e-5;

export interface VelocityOptions {
  /** K. */
  readonly temperature: number;
  /** A run records this. */
  readonly seed: number;
  /**
   * The integrator's velocity time offset: `dt/2` for both of ours.
   *
   * When non-zero, `forces` is required -- the draw is shifted by
   * `-offset F/m`. There is no default: see the note above, where assuming one
   * is what went wrong.
   */
  readonly timeOffset: number;
  /** kJ/mol/nm at `positions`, needed only when `timeOffset` is non-zero. */
  readonly forces?: Float64Array;
  /** Needed to project the velocities onto the constraint surface. */
  readonly positions: Float64Array;
  readonly split?: ConstraintSplit;
}

export function velocitiesAtTemperature(
  system: OpenMMSystem,
  options: VelocityOptions,
): Float64Array {
  const n = system.particleCount;
  const velocities = new Float64Array(n * 3);
  const kT = BOLTZMANN_KJ_PER_MOL_K * options.temperature;
  const offset = options.timeOffset ?? 0;
  if (offset !== 0 && options.forces === undefined) {
    throw new Error(
      "an integrator with a velocity time offset needs the forces at these " +
        "positions: setVelocitiesToTemperature subtracts (offset/m)F from the " +
        "Maxwell-Boltzmann draw, and leaving it out starts the run half a step " +
        "out of phase. Pass `forces`, or `timeOffset: 0` for velocity Verlet.",
    );
  }

  for (let i = 0; i < n; i++) {
    const mass = system.masses[i] as number;
    if (mass === 0) continue;
    const scale = Math.sqrt(kT / mass);
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      let value = scale * domainNormal(VELOCITY_DOMAIN, options.seed, i, axis);
      if (offset !== 0) {
        value -= (offset / mass) * ((options.forces as Float64Array)[at] as number);
      }
      velocities[at] = value;
    }
  }

  constrainVelocities(
    system,
    options.positions,
    velocities,
    VELOCITY_CONSTRAINT_TOLERANCE,
    options.split ?? splitConstraints(system),
  );
  return velocities;
}
