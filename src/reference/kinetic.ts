/**
 * Kinetic energy, in the two conventions OpenMM uses.
 *
 * Not a detail, and not a free choice. A leapfrog integrator carries its
 * velocities half a step out of step with its positions, so a kinetic energy
 * computed from them directly is evaluated at the wrong time. OpenMM therefore
 * reports kinetic energy differently depending on which integrator is running,
 * and the engine has to match it per integrator or every line of every log
 * file disagrees with the notebook's.
 *
 * From `ReferenceKernels.cpp`:
 *
 * | integrator | convention |
 * | --- | --- |
 * | `VerletIntegrator` | velocities shifted by +dt/2, then RATTLE'd at 1e-4 |
 * | `LangevinIntegrator`, `LangevinMiddleIntegrator` | the velocities as they are |
 *
 * The notebook uses `LangevinIntegrator`, so its logs are the unshifted
 * convention. The shifted one is what conserves: measured over 40 steps of
 * 1L2Y at 2 fs, the total energy ranges over 3.08e-3 kJ/mol an atom with the
 * shift and re-constraint, against 2.54e-2 without. See docs/DYNAMICS.md.
 */

import type { OpenMMSystem } from "../engine/system.js";
import { applyVelocityConstraints } from "./constraints.js";

/** The tolerance OpenMM re-constrains shifted velocities to, hard-coded there. */
export const SHIFTED_VELOCITY_TOLERANCE = 1e-4;

/** (1/2) sum m v^2, from the velocities exactly as given. */
export function kineticEnergy(
  system: OpenMMSystem,
  velocities: Float64Array,
): number {
  let total = 0;
  for (let i = 0; i < system.particleCount; i++) {
    const mass = system.masses[i] as number;
    if (mass <= 0) continue;
    const x = velocities[i * 3] as number;
    const y = velocities[i * 3 + 1] as number;
    const z = velocities[i * 3 + 2] as number;
    total += mass * (x * x + y * y + z * z);
  }
  return 0.5 * total;
}

/**
 * Velocities advanced by `timeShift`, then re-constrained -- OpenMM's
 * `computeShiftedVelocities`.
 *
 * The re-constraint is the part that is easy to leave out and matters: the
 * shift moves velocities off the constraint surface, and the components along
 * the constrained bonds then contribute kinetic energy that should not exist.
 * Leaving it out recovers only a fifth of the improvement.
 *
 * A zero shift returns the velocities untouched, with no constraint pass, which
 * is what OpenMM does.
 */
export function shiftedVelocities(
  system: OpenMMSystem,
  positions: Float64Array,
  velocities: Float64Array,
  forces: Float64Array,
  timeShift: number,
): Float64Array {
  const shifted = Float64Array.from(velocities);
  if (timeShift === 0) return shifted;
  for (let i = 0; i < system.particleCount; i++) {
    const mass = system.masses[i] as number;
    if (mass <= 0) continue;
    const scale = timeShift / mass;
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      shifted[at] = (shifted[at] as number) + (forces[at] as number) * scale;
    }
  }
  applyVelocityConstraints(
    system,
    positions,
    shifted,
    SHIFTED_VELOCITY_TOLERANCE,
  );
  return shifted;
}

/** The kinetic energy a leapfrog integrator's conserved quantity needs. */
export const shiftedKineticEnergy = (
  system: OpenMMSystem,
  positions: Float64Array,
  velocities: Float64Array,
  forces: Float64Array,
  timeShift: number,
): number =>
  kineticEnergy(
    system,
    shiftedVelocities(system, positions, velocities, forces, timeShift),
  );

/**
 * The instantaneous temperature OpenMM reports, in K.
 *
 * `2 KE / (N_dof k_B)` with the degrees of freedom reduced by every constraint
 * and by three for the centre-of-mass motion remover, which is how OpenMM's
 * `StateDataReporter` counts them.
 */
export function degreesOfFreedom(system: OpenMMSystem): number {
  let moving = 0;
  for (let i = 0; i < system.particleCount; i++) {
    if ((system.masses[i] as number) > 0) moving++;
  }
  const removesCentreOfMass = system.forces.some((f) => f.type === "CMMotionRemover");
  return 3 * moving - system.constraintCount - (removesCentreOfMass ? 3 : 0);
}
