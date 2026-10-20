/**
 * Periodic boundary conditions, as invariances rather than as an audit.
 *
 * Reading the code and concluding the minimum image is applied in the right
 * places is weaker than measuring what the engine does under the symmetries
 * periodicity implies. Three of them, and the third is the one that justifies
 * the design:
 *
 * **Moving one molecule by exactly one box vector changes nothing.** This is
 * the definitive test, because it exercises the direct-space pair sum, the
 * exclusion corrections, the reciprocal sum and the cell list all at once: the
 * molecule's neighbours are now different *atoms* reached through different
 * images, and if any of those paths failed to wrap, the energy would move by a
 * large amount.
 *
 * **Moving the whole system changes nothing.** A weaker claim -- every pair
 * separation is untouched -- but it catches anything that depends on absolute
 * position, which the PME mesh does: the charge grid is aligned to the box, so
 * a translation that is not a whole number of grid cells reshuffles which
 * B-spline weights each atom gets.
 *
 * **Moving one *atom* by one box vector changes the energy a great deal.** The
 * complement, and the reason the other two matter. Bonded terms use plain
 * differences, exactly as OpenMM's do, so a split molecule is not an
 * approximation -- it is a bond read as a box length. This measures how bad
 * that is, which is what makes "never wrap a coordinate" a rule with a number
 * behind it rather than a convention.
 */

import { parseState } from "../engine/state.js";
import { parseSystem, type OpenMMSystem } from "../engine/system.js";
import { findMolecules } from "../engine/molecules.js";
import { checkMoleculesIntact } from "../engine/wrapping.js";
import { rectangularBox } from "../reference/forces/nonbonded.js";
import { StepLoop, type RunOptions } from "./loop.js";

/**
 * Thresholds.
 *
 * The invariances are exact in exact arithmetic but not in f32: translating a
 * coordinate changes the rounding of `coordinate / extent` inside the PME
 * spread, so the mesh forces shift by about a part in 10^7. Pair forces no
 * longer contribute, since the minimum image is taken in fixed point. 1e-5
 * relative is two orders of margin over that, and three or more below what a
 * missing wrap produces -- which is not a relative error at all but a different
 * physical system.
 */
export const PBC_THRESHOLDS = {
  energyRelative: 1e-5,
  forceRelativeRms: 1e-4,
} as const;

export interface InvarianceCheck {
  readonly what: string;
  readonly energy: number;
  readonly reference: number;
  readonly energyRelativeError: number;
  /** Relative RMS change in the forces. */
  readonly forceRelativeRms: number;
  readonly ok: boolean;
}

export interface PbcComparison {
  readonly particleCount: number;
  readonly molecules: number;
  readonly box: readonly number[];
  readonly cutoff: number;
  /** Half the smallest edge: the minimum image is only meaningful below this. */
  readonly minimumImageLimit: number;
  /** Whether the fixture's own coordinates have intact molecules. */
  readonly inputIntact: boolean;
  readonly inputWorstBondedSeparation: number;
  readonly invariances: readonly InvarianceCheck[];
  /**
   * What splitting a molecule costs, as a multiple of the intact energy.
   *
   * Reported rather than bounded: the point is that it is enormous, and the
   * number says how enormous.
   */
  readonly splitMoleculeEnergyRatio: number;
  readonly thresholds: typeof PBC_THRESHOLDS;
  readonly ok: boolean;
}

export async function validatePbc(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
): Promise<PbcComparison> {
  const system: OpenMMSystem = parseSystem(texts.systemXml);
  const state = parseState(texts.stateXml);
  const box = rectangularBox(state.boxVectors);
  const molecules = findMolecules(system);
  const nonbonded = system.forces.find((f) => f.type === "NonbondedForce");
  const cutoff = nonbonded !== undefined && "cutoff" in nonbonded ? nonbonded.cutoff : 1;

  const options: RunOptions = {
    timeStep: 0.002,
    temperature: 300,
    friction: 0,
    constraintTolerance: 1e-6,
    integrator: "verlet",
    cellListInterval: 1,
  };

  const loop = await StepLoop.fromParsed(device, system, state, options);
  const invariances: InvarianceCheck[] = [];
  let splitRatio = 0;
  try {
    const base = await loop.potential(state.positions);
    const forceRms = (forces: Float64Array): number => {
      let sum = 0;
      for (const value of forces) sum += value * value;
      return Math.sqrt(sum / forces.length);
    };
    const baseRms = forceRms(base.forces);

    const compare = async (
      what: string,
      positions: Float64Array,
      expectInvariant: boolean,
    ): Promise<InvarianceCheck> => {
      const moved = await loop.potential(positions);
      let squared = 0;
      for (let i = 0; i < moved.forces.length; i++) {
        squared += ((moved.forces[i] as number) - (base.forces[i] as number)) ** 2;
      }
      const forceRelativeRms = Math.sqrt(squared / moved.forces.length) / baseRms;
      const energyRelativeError =
        Math.abs(moved.energy - base.energy) / Math.abs(base.energy);
      return {
        what,
        energy: moved.energy,
        reference: base.energy,
        energyRelativeError,
        forceRelativeRms,
        ok: expectInvariant
          ? energyRelativeError < PBC_THRESHOLDS.energyRelative &&
            forceRelativeRms < PBC_THRESHOLDS.forceRelativeRms
          : // The complement: it must *not* be invariant.
            energyRelativeError > 1e-3,
      };
    };

    // 1. One molecule, one box vector along each axis in turn. The largest
    //    molecule is the solute, which has the most neighbours to get wrong.
    let largest = 0;
    for (let m = 0; m < molecules.count; m++) {
      const size = (molecules.starts[m + 1] as number) - (molecules.starts[m] as number);
      if (size === molecules.largest) {
        largest = m;
        break;
      }
    }
    for (const axis of [0, 1, 2]) {
      const shifted = Float64Array.from(state.positions);
      for (
        let slot = molecules.starts[largest] as number;
        slot < (molecules.starts[largest + 1] as number);
        slot++
      ) {
        const atom = molecules.atoms[slot] as number;
        shifted[atom * 3 + axis] =
          (shifted[atom * 3 + axis] as number) + (box[axis] as number);
      }
      invariances.push(
        await compare(
          `the largest molecule moved one box along ${"xyz"[axis]}`,
          shifted,
          true,
        ),
      );
    }

    // 2. The whole system, by a box vector and by an arbitrary vector.
    for (const [label, offset] of [
      ["one box along x", [box[0] as number, 0, 0]],
      ["an arbitrary vector", [0.37, -1.21, 2.03]],
    ] as const) {
      const shifted = Float64Array.from(state.positions);
      for (let i = 0; i < system.particleCount; i++) {
        for (let axis = 0; axis < 3; axis++) {
          shifted[i * 3 + axis] =
            (shifted[i * 3 + axis] as number) + (offset[axis] as number);
        }
      }
      invariances.push(await compare(`the whole system moved ${label}`, shifted, true));
    }

    // 3. One atom of the largest molecule, which splits it.
    const split = Float64Array.from(state.positions);
    const victim = molecules.atoms[molecules.starts[largest] as number] as number;
    split[victim * 3] = (split[victim * 3] as number) + (box[0] as number);
    const broken = await compare(
      "one atom moved one box, splitting its molecule",
      split,
      false,
    );
    invariances.push(broken);
    splitRatio = Math.abs(broken.energy / base.energy);
  } finally {
    loop.destroy();
  }

  const inputReport = checkMoleculesIntact(system, state.positions, box);
  return {
    particleCount: system.particleCount,
    molecules: molecules.count,
    box: Array.from(box as unknown as number[]),
    cutoff,
    minimumImageLimit:
      0.5 * Math.min(box[0] as number, box[1] as number, box[2] as number),
    inputIntact: inputReport.intact,
    inputWorstBondedSeparation: inputReport.worstSeparation,
    invariances,
    splitMoleculeEnergyRatio: splitRatio,
    thresholds: PBC_THRESHOLDS,
    ok: invariances.every((check) => check.ok) && inputReport.intact,
  };
}
