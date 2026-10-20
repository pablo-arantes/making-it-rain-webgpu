/**
 * `MonteCarloBarostat`, transcribed from `MonteCarloBarostatImpl` and
 * `ReferenceMonteCarloBarostat`.
 *
 * It is not a force and it is not an integrator. It runs from
 * `updateContextState`, like `CMMotionRemover`, and what it does is a Metropolis
 * move on the volume: propose a change, rescale the box and the molecules,
 * evaluate the energy again, and accept or reject.
 *
 *     w = dE + P dV - N_molecules kT ln(V'/V)
 *     accept if w <= 0, or with probability exp(-w/kT)
 *
 * Three details that are easy to get wrong and all of them matter.
 *
 * **Molecules move, atoms do not.** Each molecule's centroid is scaled and
 * every atom in it is translated by the same offset, so internal geometry is
 * untouched and constrained bonds stay constrained. Scaling atoms individually
 * would stretch every bond by the scale factor -- a 1% volume move violates a
 * rigid water's constraints by 0.3% -- and the solver would spend every attempt
 * undoing it.
 *
 * **The `N` in the logarithm counts molecules, not atoms**, when molecules are
 * scaled rigidly, which is the default. It is the number of independent things
 * being moved, and using the atom count instead would bias the volume
 * distribution by a factor of the atoms-per-molecule ratio -- about 3 for water,
 * which is not subtle.
 *
 * **The step counter fires on the frequency-th call, not the zeroth.** OpenMM
 * does `if (++step < frequency) return; step = 0;`, so with frequency 25 the
 * first attempt is on the 25th call. `CMMotionRemover` is the opposite -- it
 * acts on step zero -- and the two are a few lines apart in a step loop.
 *
 * The acceptance rate is kept near half by adapting the proposal width every
 * ten attempts: shrink it below 25% acceptance, grow it above 75%, capped at
 * 30% of the volume.
 */

import { BOLTZMANN_KJ_PER_MOL_K } from "../engine/units.js";
import type { MonteCarloBarostat, OpenMMSystem, Vec3 } from "../engine/system.js";
import { findMolecules, type Molecules } from "../engine/molecules.js";

/**
 * Bar to kJ/mol/nm^3.
 *
 * `AVOGADRO * 1e-25`, which is OpenMM's own expression: a bar is 1e5 Pa =
 * 1e5 J/m^3, a nm^3 is 1e-27 m^3, and a mole is Avogadro's number of them.
 */
export const BAR_TO_KJ_PER_MOL_NM3 = 6.02214076e23 * 1e-25;

/** A uniform deviate in [0, 1), so a run can record its seed. */
export type Uniform = () => number;

export interface BarostatState {
  /** Calls since the last attempt. */
  step: number;
  /** nm^3, the half-width of the volume proposal. */
  volumeScale: number;
  attempted: number;
  accepted: number;
}

/** The state OpenMM starts a barostat in. */
export function initialBarostatState(boxVectors: readonly [Vec3, Vec3, Vec3]): BarostatState {
  return {
    step: 0,
    volumeScale: 0.01 * boxVolume(boxVectors),
    attempted: 0,
    accepted: 0,
  };
}

/**
 * A rectangular box's three vectors from its three edge lengths.
 *
 * The diagonal, which is easy to get wrong in a way that reads as a plausible
 * matrix: `lengths.map((l) => [l, 0, 0])` gives three *parallel* vectors, and
 * `boxVolume` then multiplies `[0][0] * [1][1] * [2][2]` to get zero. That cost
 * a round trip, and the error it produced was a cell list complaining the box
 * had shrunk to nothing rather than anything pointing at the matrix.
 */
export const diagonalBox = (lengths: Vec3): [Vec3, Vec3, Vec3] => [
  [lengths[0] as number, 0, 0],
  [0, lengths[1] as number, 0],
  [0, 0, lengths[2] as number],
];

/** The three edge lengths of a rectangular box. */
export const boxLengths = (boxVectors: readonly [Vec3, Vec3, Vec3]): Vec3 =>
  [
    boxVectors[0][0] as number,
    boxVectors[1][1] as number,
    boxVectors[2][2] as number,
  ] as unknown as Vec3;

/** The volume of a rectangular box. */
export const boxVolume = (boxVectors: readonly [Vec3, Vec3, Vec3]): number =>
  (boxVectors[0][0] as number) *
  (boxVectors[1][1] as number) *
  (boxVectors[2][2] as number);

/**
 * Whether the barostat acts on this call, advancing `state.step`.
 *
 * Mutates, like OpenMM's `++step`, because the counter is part of the move's
 * state and a caller that forgot to advance it would attempt every step.
 *
 * This is the per-step form, called once for every step as
 * `updateContextState` is. A GPU loop encodes many steps into one command
 * buffer and never sees them individually, so it uses
 * `barostatActsAfterSteps` instead.
 */
export function barostatActsNow(state: BarostatState, frequency: number): boolean {
  return barostatActsAfterSteps(state, frequency, 1);
}

/**
 * The same, for a batch of steps that has already run.
 *
 * 🔴 A step loop cannot call the per-step form. Its steps are encoded into a
 * command buffer and submitted together, so the host reaches the barostat once
 * per batch rather than once per step -- and calling `barostatActsNow` there
 * advances the counter by one instead of by however many steps ran. Measured as
 * zero attempts in a hundred steps at frequency 25.
 *
 * Any overshoot carries over rather than being dropped, so a batch that is not
 * a multiple of the frequency loses no attempts; the loop chunks to the
 * boundary anyway, which keeps the attempt on the same step OpenMM puts it on.
 */
export function barostatActsAfterSteps(
  state: BarostatState,
  frequency: number,
  steps: number,
): boolean {
  if (frequency === 0) return false;
  state.step += steps;
  if (state.step < frequency) return false;
  state.step -= frequency;
  return true;
}

/**
 * Scale every molecule's centroid, translating its atoms rigidly, in place.
 *
 * The centroid is the unweighted mean of the atoms' positions -- not the centre
 * of mass. That is OpenMM's choice and it matters for a water molecule, whose
 * centre of mass sits almost on the oxygen while its centroid does not.
 */
export function scaleMolecules(
  positions: Float64Array,
  molecules: Molecules,
  scale: number,
): void {
  for (let molecule = 0; molecule < molecules.count; molecule++) {
    const begin = molecules.starts[molecule] as number;
    const end = molecules.starts[molecule + 1] as number;
    const size = end - begin;
    const centre = [0, 0, 0];
    for (let slot = begin; slot < end; slot++) {
      const atom = molecules.atoms[slot] as number;
      for (let axis = 0; axis < 3; axis++) {
        centre[axis] = (centre[axis] as number) + (positions[atom * 3 + axis] as number);
      }
    }
    const offset = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
      const mean = (centre[axis] as number) / size;
      offset[axis] = scale * mean - mean;
    }
    for (let slot = begin; slot < end; slot++) {
      const atom = molecules.atoms[slot] as number;
      for (let axis = 0; axis < 3; axis++) {
        positions[atom * 3 + axis] =
          (positions[atom * 3 + axis] as number) + (offset[axis] as number);
      }
    }
  }
}

/** Scale a rectangular box's vectors. */
export const scaleBox = (
  boxVectors: readonly [Vec3, Vec3, Vec3],
  scale: number,
): [Vec3, Vec3, Vec3] =>
  boxVectors.map((vector) => vector.map((value) => value * scale) as unknown as Vec3) as [
    Vec3,
    Vec3,
    Vec3,
  ];

export interface BarostatAttempt {
  readonly accepted: boolean;
  /** nm^3 */
  readonly deltaVolume: number;
  readonly lengthScale: number;
  readonly initialEnergy: number;
  readonly finalEnergy: number;
  /** The Metropolis work, kJ/mol. */
  readonly work: number;
  readonly boxVectors: readonly [Vec3, Vec3, Vec3];
  readonly positions: Float64Array;
}

/**
 * One barostat attempt.
 *
 * `potential` is called twice, at the current and the proposed configuration.
 * The caller is left holding whichever was accepted; nothing here mutates its
 * inputs.
 */
export async function barostatAttempt(
  system: OpenMMSystem,
  barostat: MonteCarloBarostat,
  state: BarostatState,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  positions: Float64Array,
  potential: (
    positions: Float64Array,
    boxVectors: readonly [Vec3, Vec3, Vec3],
  ) => Promise<number>,
  uniform: Uniform,
  molecules: Molecules = findMolecules(system),
): Promise<BarostatAttempt> {
  const volume = boxVolume(boxVectors);
  const initialEnergy = await potential(positions, boxVectors);

  // Uniform in [-volumeScale, volumeScale).
  const deltaVolume = state.volumeScale * 2 * (uniform() - 0.5);
  const newVolume = volume + deltaVolume;
  const lengthScale = Math.cbrt(newVolume / volume);

  const proposedBox = scaleBox(boxVectors, lengthScale);
  const proposedPositions = Float64Array.from(positions);
  scaleMolecules(proposedPositions, molecules, lengthScale);
  const finalEnergy = await potential(proposedPositions, proposedBox);

  const pressure = barostat.defaultPressure * BAR_TO_KJ_PER_MOL_NM3;
  const kT = BOLTZMANN_KJ_PER_MOL_K * barostat.defaultTemperature;
  // Molecules, not atoms: the number of independent things being moved.
  const work =
    finalEnergy -
    initialEnergy +
    pressure * deltaVolume -
    molecules.count * kT * Math.log(newVolume / volume);

  // `w <= 0` is accepted without drawing, which is what OpenMM does -- and it
  // means the deviate stream depends on the trajectory. Reproducing a run needs
  // the same seed *and* the same sequence of accepted moves.
  const accepted = !(work > 0 && uniform() > Math.exp(-work / kT));
  state.attempted++;
  if (accepted) state.accepted++;

  if (state.attempted >= 10) {
    if (state.accepted < 0.25 * state.attempted) {
      state.volumeScale /= 1.1;
      state.attempted = 0;
      state.accepted = 0;
    } else if (state.accepted > 0.75 * state.attempted) {
      state.volumeScale = Math.min(state.volumeScale * 1.1, volume * 0.3);
      state.attempted = 0;
      state.accepted = 0;
    }
  }

  return {
    accepted,
    deltaVolume,
    lengthScale,
    initialEnergy,
    finalEnergy,
    work,
    boxVectors: accepted ? proposedBox : boxVectors,
    positions: accepted ? proposedPositions : Float64Array.from(positions),
  };
}
