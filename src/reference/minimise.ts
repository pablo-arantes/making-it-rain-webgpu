/**
 * Local energy minimisation, as OpenMM does it.
 *
 * Transcribed from `ReferenceMinimize::minimize` in
 * `platforms/reference/src/SimTKReference/ReferenceMinimize.cpp`. The algorithm
 * matters more than it looks: a minimiser's answer is wherever its objective
 * has a stationary point, so replicating OpenMM's *objective* is what makes our
 * minimum its minimum. The search itself (src/reference/lbfgs.ts) only decides
 * how fast it gets there.
 *
 * The objective is **not** the potential energy. Constraints are not imposed
 * during the search; they are turned into springs and added to the energy:
 *
 *     E(x) = U(x) + sum over constraints of (k/2) (r - d)^2
 *
 * with `k = 100 / max(1e-4, constraintTolerance)` -- so 1e6 kJ/mol/nm^2 for the
 * notebook's 1e-6 tolerance, because the 1e-4 floor wins. An outer loop
 * multiplies `k` by ten until every constrained distance is within
 * `max(1e-4, tolerance)` of its target, and a final exact projection then
 * tightens them to the tolerance the caller actually asked for.
 *
 * 🔴 Two behaviours here are surprising and are OpenMM's, not ours.
 *
 * Stiffening the springs can stop helping, and when it does OpenMM **discards
 * the whole minimisation and returns the starting positions**: the outer loop
 * resets the context before testing whether to give up. A caller that sees its
 * coordinates come back unchanged has not hit a no-op, it has hit this path, so
 * `MinimiseResult.gaveUp` says so rather than leaving it to be inferred.
 *
 * A frozen particle's gradient is zeroed rather than its force being ignored,
 * which is not the same thing once the constraint springs are added: a spring
 * between a frozen and a mobile particle still pushes the mobile one.
 */

import type { OpenMMSystem } from "../engine/system.js";
import {
  splitConstraints,
  type ConstraintSplit,
} from "../engine/constraint-clusters.js";
import { constrainPositions } from "./constraints.js";
import { lbfgs, type LbfgsResult } from "./lbfgs.js";

/** A force field: the energy at `positions`, and the forces it exerts. */
export type Potential = (
  positions: Float64Array,
) => Promise<{ energy: number; forces: Float64Array }>;

export interface MinimiseOptions {
  /**
   * kJ/mol/nm. The run stops when the root-mean-square force per particle
   * falls below this. OpenMM's default is 10.
   */
  readonly tolerance?: number;
  /** Zero means no limit, as in OpenMM. */
  readonly maxIterations?: number;
  /** The integrator's constraint tolerance, which sets the spring strength. */
  readonly constraintTolerance: number;
  readonly split?: ConstraintSplit;
}

export interface MinimiseResult {
  /** The minimised coordinates, nm. */
  readonly positions: Float64Array;
  /** kJ/mol, the potential energy alone -- the springs are not part of it. */
  readonly energy: number;
  /** kJ/mol, what the constraint springs still hold at the end. */
  readonly restraintEnergy: number;
  /** The worst relative constraint violation before the final projection. */
  readonly maxConstraintError: number;
  /** The spring strength the outer loop ended at. */
  readonly restraintStrength: number;
  /** Where it started: `100 / max(1e-4, constraintTolerance)`. */
  readonly initialRestraintStrength: number;
  /**
   * Root-mean-square objective gradient per particle, kJ/mol/nm.
   *
   * This is the quantity the tolerance is tested against, springs included, at
   * whatever `restraintStrength` the outer loop ended on -- so it is not
   * comparable across runs that took different numbers of rounds. Use
   * `potentialForceRms` to compare against a force.
   */
  readonly forceRms: number;
  /**
   * Root-mean-square potential force per particle, kJ/mol/nm, springs excluded.
   *
   * What OpenMM's `State.getForces` would report, and so the number to compare.
   * On this workflow's systems it stays close to the objective gradient anyway:
   * once the constraints hold to 1e-5 relative, `dr` is 1e-6 nm and a 1e6
   * spring contributes about 1 kJ/mol/nm.
   */
  readonly potentialForceRms: number;
  readonly rounds: number;
  readonly iterations: number;
  readonly evaluations: number;
  /** True when the outer loop abandoned the search and restored the start. */
  readonly gaveUp: boolean;
  readonly status: LbfgsResult["status"];
}

/** The worst relative violation of any constrained distance. */
export function maxConstraintError(
  system: OpenMMSystem,
  positions: Float64Array,
): number {
  let worst = 0;
  for (let c = 0; c < system.constraintCount; c++) {
    const p1 = (system.constraintP1[c] as number) * 3;
    const p2 = (system.constraintP2[c] as number) * 3;
    let squared = 0;
    for (let axis = 0; axis < 3; axis++) {
      squared += ((positions[p2 + axis] as number) - (positions[p1 + axis] as number)) ** 2;
    }
    const distance = system.constraintDistance[c] as number;
    worst = Math.max(worst, Math.abs(Math.sqrt(squared) - distance) / distance);
  }
  return worst;
}

/** The spring energy and gradient the constraints contribute. */
function addConstraintSprings(
  system: OpenMMSystem,
  positions: Float64Array,
  gradient: Float64Array,
  k: number,
): number {
  let energy = 0;
  for (let c = 0; c < system.constraintCount; c++) {
    const i = system.constraintP1[c] as number;
    const j = system.constraintP2[c] as number;
    const distance = system.constraintDistance[c] as number;
    const delta = [0, 0, 0];
    let squared = 0;
    for (let axis = 0; axis < 3; axis++) {
      const d = (positions[j * 3 + axis] as number) - (positions[i * 3 + axis] as number);
      delta[axis] = d;
      squared += d * d;
    }
    const r = Math.sqrt(squared);
    const dr = r - distance;
    const kdr = k * dr;
    energy += 0.5 * kdr * dr;
    for (let axis = 0; axis < 3; axis++) {
      const unit = (delta[axis] as number) / r;
      // The sign convention is OpenMM's: the gradient of the spring energy,
      // which pulls the pair together when `dr` is positive.
      if ((system.masses[i] as number) !== 0) {
        gradient[i * 3 + axis] = (gradient[i * 3 + axis] as number) - kdr * unit;
      }
      if ((system.masses[j] as number) !== 0) {
        gradient[j * 3 + axis] = (gradient[j * 3 + axis] as number) + kdr * unit;
      }
    }
  }
  return energy;
}

/** Root-mean-square force per particle, which is what the tolerance measures. */
export function forceRms(gradient: Float64Array, particleCount: number): number {
  let sum = 0;
  for (let i = 0; i < gradient.length; i++) sum += (gradient[i] as number) ** 2;
  return Math.sqrt(sum / particleCount);
}

export async function minimise(
  system: OpenMMSystem,
  initialPositions: Float64Array,
  potential: Potential,
  options: MinimiseOptions,
): Promise<MinimiseResult> {
  const tolerance = options.tolerance ?? 10;
  const maxIterations = options.maxIterations ?? 0;
  const split = options.split ?? splitConstraints(system);
  const workingConstraintTolerance = Math.max(1e-4, options.constraintTolerance);
  let k = 100 / workingConstraintTolerance;
  const initialRestraintStrength = k;
  const n = system.particleCount;

  // The search starts from a configuration that already satisfies the
  // constraints, to the working tolerance rather than the caller's.
  const x = Float64Array.from(initialPositions);
  constrainPositions(system, x, x, workingConstraintTolerance, split);
  const initial = Float64Array.from(x);

  // OpenMM scales the gradient tolerance by the coordinates' own magnitude, so
  // that `epsilon` in libLBFGS's `|g| / max(|x|, 1)` test becomes a root mean
  // square force. See LbfgsOptions.epsilon for the algebra.
  let meanSquare = 0;
  for (let i = 0; i < n; i++) {
    meanSquare +=
      (x[i * 3] as number) ** 2 +
      (x[i * 3 + 1] as number) ** 2 +
      (x[i * 3 + 2] as number) ** 2;
  }
  meanSquare /= n;
  const magnitude = meanSquare < 1 ? 1 : Math.sqrt(meanSquare);
  const epsilon = tolerance / magnitude;

  let lastEnergy = 0;
  let lastRestraint = 0;
  let lastGradient = new Float64Array(n * 3);
  let lastForces = new Float64Array(n * 3);
  const objective = async (
    position: Float64Array,
    gradient: Float64Array,
  ): Promise<number> => {
    const { energy, forces } = await potential(position);
    for (let i = 0; i < n; i++) {
      const frozen = (system.masses[i] as number) === 0;
      for (let axis = 0; axis < 3; axis++) {
        gradient[i * 3 + axis] = frozen ? 0 : -(forces[i * 3 + axis] as number);
      }
    }
    lastForces = Float64Array.from(forces);
    const restraint = addConstraintSprings(system, position, gradient, k);
    lastEnergy = energy;
    lastRestraint = restraint;
    lastGradient = Float64Array.from(gradient);
    return energy + restraint;
  };

  let previousError1 = 1e10;
  let previousError2 = 1e10;
  let rounds = 0;
  let iterations = 0;
  let evaluations = 0;
  let status: LbfgsResult["status"] = "converged";
  let error = 0;
  let gaveUp = false;

  for (;;) {
    rounds++;
    const result = await lbfgs(x, objective, {
      epsilon,
      maxIterations,
      // libLBFGS's defaults, which is what OpenMM leaves them at.
      memory: 6,
    });
    iterations += result.iterations;
    evaluations += result.evaluations;
    status = result.status;

    error = maxConstraintError(system, x);
    if (error <= workingConstraintTolerance) break;
    if (error >= previousError2) {
      // OpenMM resets to the starting configuration *before* this test, so
      // giving up means returning the coordinates it was given.
      x.set(initial);
      gaveUp = true;
      break;
    }
    previousError2 = previousError1;
    previousError1 = error;
    k *= 10;
    if (error > 100 * workingConstraintTolerance) {
      // Far enough from a valid configuration that the springs may not be able
      // to pull it back, so start the next round over.
      x.set(initial);
    }
  }

  if (options.constraintTolerance < workingConstraintTolerance) {
    constrainPositions(system, x, x, options.constraintTolerance, split);
  }

  return {
    positions: x,
    energy: lastEnergy,
    restraintEnergy: lastRestraint,
    maxConstraintError: error,
    restraintStrength: k,
    forceRms: forceRms(lastGradient, n),
    potentialForceRms: forceRms(lastForces, n),
    initialRestraintStrength,
    rounds,
    iterations,
    evaluations,
    gaveUp,
    status,
  };
}
