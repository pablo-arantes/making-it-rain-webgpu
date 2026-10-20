/**
 * Whether a configuration's molecules are intact.
 *
 * 🔴 The engine's bonded forces use plain differences, with no minimum image.
 * That is not an oversight -- it is what OpenMM does, because
 * `HarmonicBondForce.usesPeriodicBoundaryConditions()` is false unless a system
 * asks otherwise, and `parseSystem` rejects one that does. It is correct
 * *provided a molecule is never split across the boundary*, which holds during
 * dynamics because nothing here ever wraps a coordinate: positions grow outside
 * the box and stay there, exactly as OpenMM's do.
 *
 * It can fail on *input*. A state wrapped atom-by-atom by another tool -- some
 * PDB pipelines, some trajectory converters -- puts the two ends of a bond on
 * opposite sides of the box, and then the bonded term sees a bond a box long.
 * The result is not a small error: a 0.1 nm bond read as 3 nm, at a force
 * constant of a few hundred thousand kJ/mol/nm², is an energy of order 1e6 and
 * a force that ejects the atom on the first step.
 *
 * So this is an upload guard, not a runtime one. OpenMM's own
 * `enforcePeriodicBox=True` wraps whole *molecules* and keeps them intact, so a
 * state that came from OpenMM passes either way; it is foreign input that
 * needs checking.
 */

import type { OpenMMSystem, Vec3 } from "./system.js";

export interface BrokenPair {
  readonly kind: "bond" | "constraint";
  readonly particles: readonly [number, number];
  /** nm, as the plain difference sees it. */
  readonly separation: number;
  /** nm, what it would be under the minimum image. */
  readonly minimumImage: number;
}

export interface WrappingReport {
  /** Pairs whose plain separation exceeds half the smallest box edge. */
  readonly broken: readonly BrokenPair[];
  /** The worst plain separation found, nm. */
  readonly worstSeparation: number;
  /** Half the smallest box edge: the bound a bonded pair must stay inside. */
  readonly limit: number;
  readonly intact: boolean;
}

/**
 * Check every bonded pair and constraint.
 *
 * Half the smallest box edge is the threshold because that is where the
 * minimum image would start disagreeing with the plain difference: a pair
 * further apart than that has an image closer than itself, which no real bond
 * does.
 */
export function checkMoleculesIntact(
  system: OpenMMSystem,
  positions: Float64Array,
  box: Vec3,
): WrappingReport {
  const limit =
    0.5 * Math.min(box[0] as number, box[1] as number, box[2] as number);
  const broken: BrokenPair[] = [];
  let worstSeparation = 0;

  const check = (kind: "bond" | "constraint", a: number, b: number): void => {
    let squared = 0;
    let imageSquared = 0;
    for (let axis = 0; axis < 3; axis++) {
      const extent = box[axis] as number;
      const d = (positions[b * 3 + axis] as number) - (positions[a * 3 + axis] as number);
      squared += d * d;
      const wrapped = d - extent * Math.round(d / extent);
      imageSquared += wrapped * wrapped;
    }
    const separation = Math.sqrt(squared);
    worstSeparation = Math.max(worstSeparation, separation);
    if (separation > limit) {
      broken.push({
        kind,
        particles: [a, b],
        separation,
        minimumImage: Math.sqrt(imageSquared),
      });
    }
  };

  for (let c = 0; c < system.constraintCount; c++) {
    check("constraint", system.constraintP1[c] as number, system.constraintP2[c] as number);
  }
  for (const force of system.forces) {
    if (force.type !== "HarmonicBondForce") continue;
    for (let i = 0; i < force.count; i++) {
      check("bond", force.p1[i] as number, force.p2[i] as number);
    }
  }

  return { broken, worstSeparation, limit, intact: broken.length === 0 };
}

export class BrokenMolecules extends Error {
  constructor(readonly report: WrappingReport) {
    const worst = report.broken[0] as BrokenPair;
    super(
      `${report.broken.length} bonded pair(s) are split across the periodic ` +
        `boundary: ${worst.kind} ${worst.particles[0]}-${worst.particles[1]} is ` +
        `${worst.separation.toFixed(3)} nm apart, against ` +
        `${worst.minimumImage.toFixed(3)} nm for its nearest image and a limit ` +
        `of ${report.limit.toFixed(3)} nm (half the smallest box edge).\n\n` +
        "This configuration was wrapped atom by atom rather than molecule by " +
        "molecule. Bonded forces use plain differences -- as OpenMM's do -- so " +
        "a bond read a box long gives an energy of order 1e6 kJ/mol and a force " +
        "that ejects the atom on the first step.\n\n" +
        "OpenMM's own `enforcePeriodicBox=True` keeps molecules intact, so a " +
        "state from OpenMM does not hit this. Re-export without atom-wise " +
        "wrapping, or make the molecules whole first.",
    );
    this.name = "BrokenMolecules";
  }
}
