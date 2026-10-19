/**
 * Distance constraints: SHAKE on positions, RATTLE on velocities.
 *
 * Written to match OpenMM's `ReferenceCCMAAlgorithm::applyConstraints` term for
 * term, because the engine's whole claim is parity.  Two details of that
 * algorithm matter and neither is the obvious choice:
 *
 * **The corrections are applied simultaneously, not one constraint at a time.**
 * Every constraint's delta is computed from the same trial coordinates, and only
 * then are they all applied -- Jacobi rather than Gauss-Seidel. That is what
 * makes a per-cluster GPU kernel an exact match rather than an approximation,
 * since constraints in different clusters never interact.
 *
 * **The correction direction is the *reference* bond vector, not the trial
 * one.** `r_ij` is taken from the positions at the start of the step and stays
 * fixed through the iteration.
 *
 * CCMA's own refinement -- multiplying the deltas by an approximate inverse of
 * the constraint coupling matrix to converge faster -- is not implemented. With
 * the matrix absent this is plain SHAKE, which solves the same system and
 * reaches the same fixed point, just in more iterations. For the clusters this
 * workload produces, measured at no more than 4 atoms and 3 constraints, there
 * is very little coupling to accelerate.
 */

import type { OpenMMSystem } from "../engine/system.js";
import {
  splitConstraints,
  type ConstraintSplit,
} from "../engine/constraint-clusters.js";
import { settlePositions, settleVelocities } from "./settle.js";

/** OpenMM's default, and what it gives up after. */
export const MAX_CONSTRAINT_ITERATIONS = 150;

export interface ConstraintResult {
  readonly iterations: number;
  readonly converged: boolean;
  /** The worst relative distance error left, for reporting. */
  readonly worstRelativeError: number;
}

/** Inverse masses, with zero for a frozen particle. */
export function inverseMasses(system: OpenMMSystem): Float64Array {
  const inverse = new Float64Array(system.particleCount);
  for (let i = 0; i < system.particleCount; i++) {
    const mass = system.masses[i] as number;
    inverse[i] = mass === 0 ? 0 : 1 / mass;
  }
  return inverse;
}

interface Prepared {
  readonly referenceVector: Float64Array;
  readonly referenceLengthSquared: Float64Array;
  readonly reducedMass: Float64Array;
}

/**
 * The per-constraint quantities that stay fixed through the iteration.
 *
 * `reducedMass` is OpenMM's `0.5/(invM_i + invM_j)`, which carries the factor of
 * a half that the delta would otherwise need.
 */
function prepare(
  system: OpenMMSystem,
  referencePositions: Float64Array,
  inverse: Float64Array,
  indices?: Int32Array,
): Prepared {
  const count = indices === undefined ? system.constraintCount : indices.length;
  const at = (k: number): number => (indices === undefined ? k : (indices[k] as number));
  const referenceVector = new Float64Array(count * 3);
  const referenceLengthSquared = new Float64Array(count);
  const reducedMass = new Float64Array(count);
  for (let k = 0; k < count; k++) {
    const c = at(k);
    const i = (system.constraintP1[c] as number) * 3;
    const j = (system.constraintP2[c] as number) * 3;
    const x = (referencePositions[i] as number) - (referencePositions[j] as number);
    const y = (referencePositions[i + 1] as number) - (referencePositions[j + 1] as number);
    const z = (referencePositions[i + 2] as number) - (referencePositions[j + 2] as number);
    referenceVector[k * 3] = x;
    referenceVector[k * 3 + 1] = y;
    referenceVector[k * 3 + 2] = z;
    referenceLengthSquared[k] = x * x + y * y + z * z;
    const sum =
      (inverse[system.constraintP1[c] as number] as number) +
      (inverse[system.constraintP2[c] as number] as number);
    reducedMass[k] = sum === 0 ? 0 : 0.5 / sum;
  }
  return { referenceVector, referenceLengthSquared, reducedMass };
}

function applyDeltas(
  system: OpenMMSystem,
  trial: Float64Array,
  inverse: Float64Array,
  prepared: Prepared,
  delta: Float64Array,
  indices?: Int32Array,
): void {
  const count = indices === undefined ? system.constraintCount : indices.length;
  for (let k = 0; k < count; k++) {
    const c = indices === undefined ? k : (indices[k] as number);
    const scale = delta[k] as number;
    if (scale === 0) continue;
    const p1 = system.constraintP1[c] as number;
    const p2 = system.constraintP2[c] as number;
    const invI = inverse[p1] as number;
    const invJ = inverse[p2] as number;
    for (let axis = 0; axis < 3; axis++) {
      const dr = (prepared.referenceVector[k * 3 + axis] as number) * scale;
      trial[p1 * 3 + axis] = (trial[p1 * 3 + axis] as number) + dr * invI;
      trial[p2 * 3 + axis] = (trial[p2 * 3 + axis] as number) - dr * invJ;
    }
  }
}

/**
 * SHAKE: project `trialPositions` onto the constraint surface.
 *
 * `referencePositions` are the positions at the start of the step, which both
 * define the correction directions and are assumed already to satisfy the
 * constraints. `trialPositions` is modified in place.
 *
 * `tolerance` is relative, as OpenMM's is: a constraint is satisfied when its
 * squared length lies within `[(1-t)^2, (1+t)^2]` of the target squared length.
 */
export function applyPositionConstraints(
  system: OpenMMSystem,
  referencePositions: Float64Array,
  trialPositions: Float64Array,
  tolerance: number,
  maxIterations = MAX_CONSTRAINT_ITERATIONS,
  subset?: Int32Array,
): ConstraintResult {
  const indices = subset;
  const count = indices === undefined ? system.constraintCount : indices.length;
  if (count === 0) return { iterations: 0, converged: true, worstRelativeError: 0 };
  const inverse = inverseMasses(system);
  const prepared = prepare(system, referencePositions, inverse, indices);
  const delta = new Float64Array(count);
  const lowerTolerance = 1 - 2 * tolerance + tolerance * tolerance;
  const upperTolerance = 1 + 2 * tolerance + tolerance * tolerance;

  let iterations = 0;
  let worstRelativeError = 0;
  while (iterations < maxIterations) {
    let converged = 0;
    worstRelativeError = 0;
    for (let k = 0; k < count; k++) {
      const c = indices === undefined ? k : (indices[k] as number);
      const p1 = (system.constraintP1[c] as number) * 3;
      const p2 = (system.constraintP2[c] as number) * 3;
      const x = (trialPositions[p1] as number) - (trialPositions[p2] as number);
      const y = (trialPositions[p1 + 1] as number) - (trialPositions[p2 + 1] as number);
      const z = (trialPositions[p1 + 2] as number) - (trialPositions[p2 + 2] as number);
      const trialLengthSquared = x * x + y * y + z * z;
      const target = system.constraintDistance[c] as number;
      const targetSquared = target * target;
      const difference = targetSquared - trialLengthSquared;
      const projection =
        x * (prepared.referenceVector[k * 3] as number) +
        y * (prepared.referenceVector[k * 3 + 1] as number) +
        z * (prepared.referenceVector[k * 3 + 2] as number);
      delta[k] = projection === 0 ? 0 : ((prepared.reducedMass[k] as number) * difference) / projection;
      worstRelativeError = Math.max(
        worstRelativeError,
        Math.abs(Math.sqrt(trialLengthSquared) / target - 1),
      );
      if (
        trialLengthSquared >= lowerTolerance * targetSquared &&
        trialLengthSquared <= upperTolerance * targetSquared
      ) {
        converged++;
      }
    }
    if (converged === count) {
      return { iterations, converged: true, worstRelativeError };
    }
    iterations++;
    applyDeltas(system, trialPositions, inverse, prepared, delta, indices);
  }
  return { iterations, converged: false, worstRelativeError };
}

/**
 * RATTLE: project `velocities` so that every constrained distance stops
 * changing, which means the relative velocity along each bond is zero.
 *
 * `positions` are the constrained positions the velocities belong to.
 * `velocities` is modified in place.
 */
export function applyVelocityConstraints(
  system: OpenMMSystem,
  positions: Float64Array,
  velocities: Float64Array,
  tolerance: number,
  maxIterations = MAX_CONSTRAINT_ITERATIONS,
  subset?: Int32Array,
): ConstraintResult {
  const indices = subset;
  const count = indices === undefined ? system.constraintCount : indices.length;
  if (count === 0) return { iterations: 0, converged: true, worstRelativeError: 0 };
  const inverse = inverseMasses(system);
  const prepared = prepare(system, positions, inverse, indices);
  const delta = new Float64Array(count);

  let iterations = 0;
  let worst = 0;
  while (iterations < maxIterations) {
    let converged = 0;
    worst = 0;
    for (let k = 0; k < count; k++) {
      const c = indices === undefined ? k : (indices[k] as number);
      const p1 = (system.constraintP1[c] as number) * 3;
      const p2 = (system.constraintP2[c] as number) * 3;
      const projection =
        ((velocities[p1] as number) - (velocities[p2] as number)) *
          (prepared.referenceVector[k * 3] as number) +
        ((velocities[p1 + 1] as number) - (velocities[p2 + 1] as number)) *
          (prepared.referenceVector[k * 3 + 1] as number) +
        ((velocities[p1 + 2] as number) - (velocities[p2 + 2] as number)) *
          (prepared.referenceVector[k * 3 + 2] as number);
      const lengthSquared = prepared.referenceLengthSquared[k] as number;
      // Unlike the position case this divides by the *reference* squared
      // length, not by the target distance squared.
      delta[k] =
        lengthSquared === 0
          ? 0
          : (-2 * (prepared.reducedMass[k] as number) * projection) / lengthSquared;
      worst = Math.max(worst, Math.abs(delta[k] as number));
      if (Math.abs(delta[k] as number) <= tolerance) converged++;
    }
    if (converged === count) {
      return { iterations, converged: true, worstRelativeError: worst };
    }
    iterations++;
    applyDeltas(system, velocities, inverse, prepared, delta, indices);
  }
  return { iterations, converged: false, worstRelativeError: worst };
}


/**
 * Apply every constraint, routing each cluster to the solver OpenMM routes it
 * to: SETTLE for the closed three-atom triangles, the iterative solver for the
 * rest.
 *
 * `split` can be passed in to avoid reclassifying on every step; it depends only
 * on the system, never on the coordinates.
 */
export function constrainPositions(
  system: OpenMMSystem,
  referencePositions: Float64Array,
  trialPositions: Float64Array,
  tolerance: number,
  split: ConstraintSplit = splitConstraints(system),
  maxIterations = MAX_CONSTRAINT_ITERATIONS,
): ConstraintResult {
  settlePositions(system, split.settle, referencePositions, trialPositions);
  return applyPositionConstraints(
    system,
    referencePositions,
    trialPositions,
    tolerance,
    maxIterations,
    split.iterative,
  );
}

/** The same routing, for velocities. */
export function constrainVelocities(
  system: OpenMMSystem,
  positions: Float64Array,
  velocities: Float64Array,
  tolerance: number,
  split: ConstraintSplit = splitConstraints(system),
  maxIterations = MAX_CONSTRAINT_ITERATIONS,
): ConstraintResult {
  settleVelocities(system, split.settle, positions, velocities);
  return applyVelocityConstraints(
    system,
    positions,
    velocities,
    tolerance,
    maxIterations,
    split.iterative,
  );
}
