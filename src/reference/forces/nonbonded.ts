/**
 * Real-space (direct) Ewald and Lennard-Jones, in double precision.
 *
 * The decomposition here is not inferred, it is measured.
 * `tools/build_nonbonded_fixture.py` reproduces OpenMM's own force-group split
 * from first principles and agrees with it to 2e-9 kJ/mol:
 *
 *   direct group = pair erfc Coulomb + pair LJ            (within the cutoff,
 *                                                          excluded pairs skipped)
 *                + exclusion correction                   (every exception)
 *                + exception interactions                 (every exception)
 *                + the analytic dispersion correction
 *
 *   reciprocal group = the reciprocal sum + the self-energy term
 *
 * The exclusion correction is the part that is easy to miss and impossible to
 * ignore: reciprocal space sums over *every* pair, including the ones the force
 * field excludes, so each excluded pair's reciprocal contribution has to be
 * taken back out in real space. For the 3HTB fixture those corrections are
 * +1.67e6 kJ/mol, which is most of what the direct group reports -- a direct
 * group that looked like a plain pair sum would be wrong by more than ten times
 * the total energy.
 *
 * Combination rules are OpenMM's `NonbondedForce` ones: sigma is the arithmetic
 * mean, epsilon the geometric mean.
 */

import { ONE_4PI_EPS0 } from "../../engine/units.js";
import type { NonbondedForce, Vec3 } from "../../engine/system.js";
import { erf, erfc, erfDerivative } from "../erf.js";

export class TriclinicBoxUnsupported extends Error {
  constructor(vectors: readonly Vec3[]) {
    super(
      "the reference nonbonded implementation handles rectangular boxes only, " +
        `and these vectors are triclinic: ${JSON.stringify(vectors)}. ` +
        "Modeller.addSolvent defaults to a cubic box, which is what the " +
        "notebook produces; a triclinic box needs the general minimum image.",
    );
    this.name = "TriclinicBoxUnsupported";
  }
}

/** The box edge lengths, after checking the box really is rectangular. */
export function rectangularBox(vectors: readonly [Vec3, Vec3, Vec3]): Vec3 {
  const offDiagonal = [
    vectors[0][1], vectors[0][2],
    vectors[1][0], vectors[1][2],
    vectors[2][0], vectors[2][1],
  ];
  if (offDiagonal.some((v) => Math.abs(v) > 1e-12)) throw new TriclinicBoxUnsupported(vectors);
  return [vectors[0][0], vectors[1][1], vectors[2][2]];
}

export interface DirectSpaceEnergy {
  /** erfc-screened Coulomb over non-excluded pairs inside the cutoff. */
  readonly coulomb: number;
  /** Lennard-Jones over non-excluded pairs inside the cutoff. */
  readonly lennardJones: number;
  /** Minus the reciprocal-space contribution of every excluded pair. */
  readonly exclusionCorrection: number;
  /** The exceptions' own unscreened Coulomb and LJ. */
  readonly exceptions: number;
  /** The analytic long-range dispersion correction; constant at fixed volume. */
  readonly dispersion: number;
  /** The sum, which is what OpenMM reports for the force's own group. */
  readonly total: number;
  readonly pairsWithinCutoff: number;
}

/**
 * OpenMM's analytic long-range dispersion correction.
 *
 *   E = 8 pi N^2 / V * ( <eps sigma^12> / (9 rc^9) - <eps sigma^6> / (3 rc^3) )
 *
 * The averages run over all *unordered pairs including i == j*, so N(N+1)/2 of
 * them, with the force field's combination rules applied to each. Determined by
 * measurement against OpenMM, which it matches to nine decimal places: see
 * tools/build_nonbonded_fixture.py. It depends on the volume but not on any
 * coordinate, so it contributes no force -- only energy, and pressure.
 */
export function dispersionCorrection(
  force: NonbondedForce,
  boxVolume: number,
): number {
  const n = force.particleCount;
  if (n === 0) return 0;

  // Grouped by distinct (sigma, epsilon) rather than looped over every pair:
  // the average is the same, and N(N+1)/2 is 3.6e8 pairs for the solvated
  // protein-ligand fixture against a few thousand class pairs. OpenMM groups
  // the same way.
  const classIndex = new Map<string, number>();
  const sigmas: number[] = [];
  const epsilons: number[] = [];
  const counts: number[] = [];
  for (let i = 0; i < n; i++) {
    const sigma = force.sigma[i] as number;
    const epsilon = force.epsilon[i] as number;
    const key = `${sigma},${epsilon}`;
    let index = classIndex.get(key);
    if (index === undefined) {
      index = sigmas.length;
      classIndex.set(key, index);
      sigmas.push(sigma);
      epsilons.push(epsilon);
      counts.push(0);
    }
    counts[index] = (counts[index] as number) + 1;
  }

  let sigma6Sum = 0;
  let sigma12Sum = 0;
  const classCount = sigmas.length;
  for (let c = 0; c < classCount; c++) {
    for (let d = c; d < classCount; d++) {
      const sigma = 0.5 * ((sigmas[c] as number) + (sigmas[d] as number));
      const epsilon = Math.sqrt((epsilons[c] as number) * (epsilons[d] as number));
      const sigma6 = sigma ** 6;
      // Unordered pairs *including* i == j, which is what OpenMM averages over:
      // n(n+1)/2 within a class, n_c n_d between two.
      const nc = counts[c] as number;
      const multiplicity =
        c === d ? (nc * (nc + 1)) / 2 : nc * (counts[d] as number);
      sigma6Sum += multiplicity * epsilon * sigma6;
      sigma12Sum += multiplicity * epsilon * sigma6 * sigma6;
    }
  }
  const pairs = (n * (n + 1)) / 2;
  const cutoff = force.cutoff;
  const average6 = sigma6Sum / pairs;
  const average12 = sigma12Sum / pairs;
  return (
    ((8 * Math.PI * n * n) / boxVolume) *
    (average12 / (9 * cutoff ** 9) - average6 / (3 * cutoff ** 3))
  );
}

/**
 * The same correction by brute force over every pair, for the test that holds
 * the grouped version to it.
 */
export function dispersionCorrectionNaive(
  force: NonbondedForce,
  boxVolume: number,
): number {
  const n = force.particleCount;
  if (n === 0) return 0;
  let sigma6Sum = 0;
  let sigma12Sum = 0;
  let pairs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const sigma = 0.5 * ((force.sigma[i] as number) + (force.sigma[j] as number));
      const epsilon = Math.sqrt(
        (force.epsilon[i] as number) * (force.epsilon[j] as number),
      );
      const sigma6 = sigma ** 6;
      sigma6Sum += epsilon * sigma6;
      sigma12Sum += epsilon * sigma6 * sigma6;
      pairs++;
    }
  }
  const cutoff = force.cutoff;
  return (
    ((8 * Math.PI * n * n) / boxVolume) *
    (sigma12Sum / pairs / (9 * cutoff ** 9) - sigma6Sum / pairs / (3 * cutoff ** 3))
  );
}

/** The set of excluded pairs, keyed so a pair can be looked up in O(1). */
function exclusionSet(force: NonbondedForce): Set<number> {
  const set = new Set<number>();
  for (let e = 0; e < force.exceptionCount; e++) {
    const a = force.exceptionP1[e] as number;
    const b = force.exceptionP2[e] as number;
    // Particle indices fit in 21 bits for any system this engine can run, and
    // a 42-bit key stays an exact double.
    set.add(Math.min(a, b) * 2097152 + Math.max(a, b));
  }
  return set;
}

interface PairContext {
  readonly alpha: number;
  readonly cutoffSquared: number;
  readonly box: Vec3;
  readonly positions: Float64Array;
  readonly force: NonbondedForce;
  readonly forces?: Float64Array | undefined;
}

/** One non-excluded pair: erfc Coulomb plus LJ, with its force. */
function pairInteraction(context: PairContext, i: number, j: number): [number, number] {
  const { positions, box, force, alpha, forces } = context;
  const ix = i * 3;
  const jx = j * 3;
  let dx = (positions[jx] as number) - (positions[ix] as number);
  let dy = (positions[jx + 1] as number) - (positions[ix + 1] as number);
  let dz = (positions[jx + 2] as number) - (positions[ix + 2] as number);
  dx -= (box[0] as number) * Math.round(dx / (box[0] as number));
  dy -= (box[1] as number) * Math.round(dy / (box[1] as number));
  dz -= (box[2] as number) * Math.round(dz / (box[2] as number));
  const rSquared = dx * dx + dy * dy + dz * dz;
  if (rSquared >= context.cutoffSquared) return [0, 0];
  const r = Math.sqrt(rSquared);

  const chargeProduct =
    ONE_4PI_EPS0 * (force.charge[i] as number) * (force.charge[j] as number);
  const alphaR = alpha * r;
  const coulomb = (chargeProduct * erfc(alphaR)) / r;
  // d/dr [erfc(a r)/r] = -erfc(a r)/r^2 - a erf'(a r)/r
  let dEdR = chargeProduct * (-erfc(alphaR) / rSquared - (alpha * erfDerivative(alphaR)) / r);

  let lennardJones = 0;
  const epsilon = Math.sqrt(
    (force.epsilon[i] as number) * (force.epsilon[j] as number),
  );
  if (epsilon !== 0) {
    const sigma = 0.5 * ((force.sigma[i] as number) + (force.sigma[j] as number));
    const sigmaOverR6 = (sigma * sigma / rSquared) ** 3;
    const sigmaOverR12 = sigmaOverR6 * sigmaOverR6;
    lennardJones = 4 * epsilon * (sigmaOverR12 - sigmaOverR6);
    dEdR += ((4 * epsilon) / r) * (6 * sigmaOverR6 - 12 * sigmaOverR12);
  }

  if (forces !== undefined) {
    // F_i = (dE/dr / r) * d, and the pair is equal and opposite.
    const scale = dEdR / r;
    forces[ix] = (forces[ix] as number) + scale * dx;
    forces[ix + 1] = (forces[ix + 1] as number) + scale * dy;
    forces[ix + 2] = (forces[ix + 2] as number) + scale * dz;
    forces[jx] = (forces[jx] as number) - scale * dx;
    forces[jx + 1] = (forces[jx + 1] as number) - scale * dy;
    forces[jx + 2] = (forces[jx + 2] as number) - scale * dz;
  }
  return [coulomb, lennardJones];
}

/**
 * The exclusion corrections and the exceptions' own interactions.
 *
 * Both run over every exception regardless of the cutoff: an excluded pair's
 * reciprocal contribution exists whatever the distance, and OpenMM applies an
 * exception's own parameters unconditionally too.
 */
function exceptionTerms(
  context: PairContext,
): { exclusionCorrection: number; exceptions: number } {
  const { positions, box, force, alpha, forces } = context;
  let exclusionCorrection = 0;
  let exceptions = 0;
  for (let e = 0; e < force.exceptionCount; e++) {
    const i = force.exceptionP1[e] as number;
    const j = force.exceptionP2[e] as number;
    const ix = i * 3;
    const jx = j * 3;
    let dx = (positions[jx] as number) - (positions[ix] as number);
    let dy = (positions[jx + 1] as number) - (positions[ix + 1] as number);
    let dz = (positions[jx + 2] as number) - (positions[ix + 2] as number);
    dx -= (box[0] as number) * Math.round(dx / (box[0] as number));
    dy -= (box[1] as number) * Math.round(dy / (box[1] as number));
    dz -= (box[2] as number) * Math.round(dz / (box[2] as number));
    const rSquared = dx * dx + dy * dy + dz * dz;
    const r = Math.sqrt(rSquared);
    const alphaR = alpha * r;
    let dEdR = 0;

    // Take back the reciprocal-space contribution of this excluded pair, using
    // the particles' own charges rather than the exception's chargeProd.
    const particleProduct =
      ONE_4PI_EPS0 * (force.charge[i] as number) * (force.charge[j] as number);
    if (particleProduct !== 0) {
      exclusionCorrection -= (particleProduct * erf(alphaR)) / r;
      // d/dr [-erf(a r)/r] = erf(a r)/r^2 - a erf'(a r)/r
      dEdR += particleProduct * (erf(alphaR) / rSquared - (alpha * erfDerivative(alphaR)) / r);
    }

    // Then the exception's own interaction, unscreened.
    const chargeProduct = ONE_4PI_EPS0 * (force.exceptionChargeProd[e] as number);
    if (chargeProduct !== 0) {
      exceptions += chargeProduct / r;
      dEdR += -chargeProduct / rSquared;
    }
    const epsilon = force.exceptionEpsilon[e] as number;
    if (epsilon !== 0) {
      const sigma = force.exceptionSigma[e] as number;
      const sigmaOverR6 = (sigma * sigma / rSquared) ** 3;
      const sigmaOverR12 = sigmaOverR6 * sigmaOverR6;
      exceptions += 4 * epsilon * (sigmaOverR12 - sigmaOverR6);
      dEdR += ((4 * epsilon) / r) * (6 * sigmaOverR6 - 12 * sigmaOverR12);
    }

    if (forces !== undefined && dEdR !== 0) {
      const scale = dEdR / r;
      forces[ix] = (forces[ix] as number) + scale * dx;
      forces[ix + 1] = (forces[ix + 1] as number) + scale * dy;
      forces[ix + 2] = (forces[ix + 2] as number) + scale * dz;
      forces[jx] = (forces[jx] as number) - scale * dx;
      forces[jx + 1] = (forces[jx + 1] as number) - scale * dy;
      forces[jx + 2] = (forces[jx + 2] as number) - scale * dz;
    }
  }
  return { exclusionCorrection, exceptions };
}

/**
 * Direct space by brute force: every pair, minimum image.
 *
 * O(N^2) and therefore only for small systems, but it depends on nothing but
 * the pair loop, which makes it the thing `directSpace` is checked against.
 */
export function directSpaceNaive(
  force: NonbondedForce,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  forces?: Float64Array,
): DirectSpaceEnergy {
  const box = rectangularBox(boxVectors);
  const context: PairContext = {
    alpha,
    cutoffSquared: force.cutoff * force.cutoff,
    box,
    positions,
    force,
    forces,
  };
  const excluded = exclusionSet(force);
  let coulomb = 0;
  let lennardJones = 0;
  let pairsWithinCutoff = 0;
  const n = force.particleCount;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (excluded.has(i * 2097152 + j)) continue;
      const [c, l] = pairInteraction(context, i, j);
      if (c !== 0 || l !== 0) pairsWithinCutoff++;
      coulomb += c;
      lennardJones += l;
    }
  }
  const { exclusionCorrection, exceptions } = exceptionTerms(context);
  const dispersion = force.useDispersionCorrection
    ? dispersionCorrection(force, box[0] * box[1] * box[2])
    : 0;
  return {
    coulomb,
    lennardJones,
    exclusionCorrection,
    exceptions,
    dispersion,
    total: coulomb + lennardJones + exclusionCorrection + exceptions + dispersion,
    pairsWithinCutoff,
  };
}

/**
 * Direct space with a cell list.
 *
 * Same physics as `directSpaceNaive`, which is what
 * `test/nonbonded-reference.test.ts` holds it to on the systems where both can
 * run. Cells are at least one cutoff wide, so a pair can only be in one of the
 * 27 cells around an atom's own.
 */
export function directSpace(
  force: NonbondedForce,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  forces?: Float64Array,
): DirectSpaceEnergy {
  const box = rectangularBox(boxVectors);
  const cutoff = force.cutoff;
  const n = force.particleCount;
  const counts: [number, number, number] = [
    Math.max(1, Math.floor((box[0] as number) / cutoff)),
    Math.max(1, Math.floor((box[1] as number) / cutoff)),
    Math.max(1, Math.floor((box[2] as number) / cutoff)),
  ];
  // Fewer than three cells on an axis means the 27-cell scan would visit the
  // same cell twice and double-count; fall back to the honest O(N^2) loop.
  if (counts[0] < 3 || counts[1] < 3 || counts[2] < 3) {
    return directSpaceNaive(force, positions, boxVectors, alpha, forces);
  }

  const cellCount = counts[0] * counts[1] * counts[2];
  const cellOf = new Int32Array(n);
  const population = new Int32Array(cellCount + 1);
  const wrapped = (value: number, extent: number, divisions: number): number => {
    const fraction = value / extent - Math.floor(value / extent);
    const index = Math.floor(fraction * divisions);
    return index === divisions ? divisions - 1 : index;
  };
  for (let i = 0; i < n; i++) {
    const cx = wrapped(positions[i * 3] as number, box[0] as number, counts[0]);
    const cy = wrapped(positions[i * 3 + 1] as number, box[1] as number, counts[1]);
    const cz = wrapped(positions[i * 3 + 2] as number, box[2] as number, counts[2]);
    const cell = (cx * counts[1] + cy) * counts[2] + cz;
    cellOf[i] = cell;
    population[cell + 1] = (population[cell + 1] as number) + 1;
  }
  for (let c = 0; c < cellCount; c++) {
    population[c + 1] = (population[c + 1] as number) + (population[c] as number);
  }
  const order = new Int32Array(n);
  const cursor = Int32Array.from(population.subarray(0, cellCount));
  for (let i = 0; i < n; i++) {
    const cell = cellOf[i] as number;
    order[cursor[cell] as number] = i;
    cursor[cell] = (cursor[cell] as number) + 1;
  }

  const context: PairContext = {
    alpha,
    cutoffSquared: cutoff * cutoff,
    box,
    positions,
    force,
    forces,
  };
  const excluded = exclusionSet(force);
  let coulomb = 0;
  let lennardJones = 0;
  let pairsWithinCutoff = 0;

  for (let cx = 0; cx < counts[0]; cx++) {
    for (let cy = 0; cy < counts[1]; cy++) {
      for (let cz = 0; cz < counts[2]; cz++) {
        const cell = (cx * counts[1] + cy) * counts[2] + cz;
        const start = population[cell] as number;
        const end = population[cell + 1] as number;
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dz = -1; dz <= 1; dz++) {
              const nx = (cx + dx + counts[0]) % counts[0];
              const ny = (cy + dy + counts[1]) % counts[1];
              const nz = (cz + dz + counts[2]) % counts[2];
              const neighbour = (nx * counts[1] + ny) * counts[2] + nz;
              // Each unordered cell pair once; within a cell, each pair once.
              if (neighbour < cell) continue;
              const nStart = population[neighbour] as number;
              const nEnd = population[neighbour + 1] as number;
              for (let a = start; a < end; a++) {
                const i = order[a] as number;
                const from = neighbour === cell ? a + 1 : nStart;
                for (let b = from; b < nEnd; b++) {
                  const j = order[b] as number;
                  const lo = i < j ? i : j;
                  const hi = i < j ? j : i;
                  if (excluded.has(lo * 2097152 + hi)) continue;
                  const [c, l] = pairInteraction(context, lo, hi);
                  if (c !== 0 || l !== 0) pairsWithinCutoff++;
                  coulomb += c;
                  lennardJones += l;
                }
              }
            }
          }
        }
      }
    }
  }

  const { exclusionCorrection, exceptions } = exceptionTerms(context);
  const dispersion = force.useDispersionCorrection
    ? dispersionCorrection(force, (box[0] as number) * (box[1] as number) * (box[2] as number))
    : 0;
  return {
    coulomb,
    lennardJones,
    exclusionCorrection,
    exceptions,
    dispersion,
    total: coulomb + lennardJones + exclusionCorrection + exceptions + dispersion,
    pairsWithinCutoff,
  };
}

/**
 * The correction a non-neutral system needs, which OpenMM also reports in the
 * reciprocal group.
 *
 *   E = -pi Q^2 / (2 alpha^2 V) * ONE_4PI_EPS0
 *
 * Ewald summation implicitly immerses the cell in a neutralising background,
 * and this is that background's self-interaction. Determined by measurement:
 * sweeping the net charge from 0 to 2 e and comparing OpenMM's reciprocal group
 * against the exact k-space sum plus the self energy recovers this expression,
 * and with it in place our PME matches OpenMM's to 2e-7 relative.
 *
 * `addSolvent` neutralises, so for anything the app builds this is essentially
 * zero -- but it is not exactly zero once a ligand's charges have been rounded,
 * and a user can upload whatever system they like. It depends on no coordinate,
 * so it contributes no force.
 */
export function netChargeCorrection(
  force: NonbondedForce,
  alpha: number,
  boxVolume: number,
): number {
  let netCharge = 0;
  for (let i = 0; i < force.particleCount; i++) {
    netCharge += force.charge[i] as number;
  }
  return (
    (-Math.PI * netCharge * netCharge * ONE_4PI_EPS0) /
    (2 * alpha * alpha * boxVolume)
  );
}

/** The Ewald self-energy term, which OpenMM reports in the reciprocal group. */
export function selfEnergy(force: NonbondedForce, alpha: number): number {
  let chargeSquaredSum = 0;
  for (let i = 0; i < force.particleCount; i++) {
    const q = force.charge[i] as number;
    chargeSquaredSum += q * q;
  }
  return (-ONE_4PI_EPS0 * alpha * chargeSquaredSum) / Math.sqrt(Math.PI);
}
