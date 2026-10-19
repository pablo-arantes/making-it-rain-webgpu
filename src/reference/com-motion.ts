/**
 * Centre-of-mass motion removal.
 *
 * `CMMotionRemover` is in the `System` our force fields build -- with
 * `frequency = 1`, so it runs **every step** -- and it is not a force. OpenMM
 * applies it from `ContextImpl::updateContextState`, which an integrator calls
 * at the start of a step, before the forces are computed. It subtracts the
 * mass-weighted mean velocity from every particle.
 *
 * 🔴 Leaving it out is not a small error, and it hides well. It shifts every
 * atom by the same amount, so every constrained distance stays satisfied and
 * every constraint test still passes -- the trajectory just slides. Measured on
 * one 2 fs step: without it our positions sit 1.75e-5 nm from OpenMM's on 1L2Y
 * and 5.2e-6 on 3HTB; with it, 5.7e-9 on both, which is f64 round-off through
 * the constraint solve.
 *
 * It was also the reason a residual got attributed to the wrong cause twice.
 * See docs/DYNAMICS.md.
 */

import type { CMMotionRemover, OpenMMSystem } from "../engine/system.js";

/** The remover a system carries, if it has one. */
export function centreOfMassRemover(
  system: OpenMMSystem,
): CMMotionRemover | undefined {
  return system.forces.find(
    (f): f is CMMotionRemover => f.type === "CMMotionRemover",
  );
}

/**
 * Whether the remover acts on this step.
 *
 * OpenMM's test is `stepCount % frequency != 0 -> return`, so it acts on step
 * zero and every `frequency`-th step after it.
 */
export const removesOnStep = (frequency: number, stepCount: number): boolean =>
  frequency > 0 && stepCount % frequency === 0;

/**
 * Subtract the mass-weighted mean velocity, in place.
 *
 * The momentum sum runs over every particle and the total mass with it, so a
 * frozen particle -- mass zero -- contributes to neither, and keeps its
 * velocity.
 */
export function removeCentreOfMassMotion(
  system: OpenMMSystem,
  velocities: Float64Array,
): readonly [number, number, number] {
  let totalMass = 0;
  let px = 0;
  let py = 0;
  let pz = 0;
  for (let i = 0; i < system.particleCount; i++) {
    const mass = system.masses[i] as number;
    totalMass += mass;
    px += mass * (velocities[i * 3] as number);
    py += mass * (velocities[i * 3 + 1] as number);
    pz += mass * (velocities[i * 3 + 2] as number);
  }
  if (totalMass === 0) return [0, 0, 0];
  const vx = px / totalMass;
  const vy = py / totalMass;
  const vz = pz / totalMass;
  for (let i = 0; i < system.particleCount; i++) {
    if ((system.masses[i] as number) === 0) continue;
    velocities[i * 3] = (velocities[i * 3] as number) - vx;
    velocities[i * 3 + 1] = (velocities[i * 3 + 1] as number) - vy;
    velocities[i * 3 + 2] = (velocities[i * 3 + 2] as number) - vz;
  }
  return [vx, vy, vz];
}
